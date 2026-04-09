/**
 * MapAgent 交通流可视化引擎 v2
 * 使用 HTML5 Canvas 进行高性能渲染
 * 完整功能：地图 + 交通流 + 路径规划 + LLM 对话
 */

// ===== 工具函数 =====
const Utils = {
    parseColor(color) {
        const hex = color.replace('#', '');
        return {
            r: parseInt(hex.substr(0, 2), 16),
            g: parseInt(hex.substr(2, 2), 16),
            b: parseInt(hex.substr(4, 2), 16)
        };
    },

    worldToScreen(worldX, worldY, view) {
        return {
            x: (worldX - view.centerX) * view.zoom + view.width / 2,
            y: (worldY - view.centerY) * view.zoom + view.height / 2
        };
    },

    screenToWorld(screenX, screenY, view) {
        return {
            x: (screenX - view.width / 2) / view.zoom + view.centerX,
            y: (screenY - view.height / 2) / view.zoom + view.centerY
        };
    },

    fitViewToBounds(bounds, width, height, padding = 50) {
        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;
        const spanX = bounds.maxX - bounds.minX + padding * 2;
        const spanY = bounds.maxY - bounds.minY + padding * 2;
        const zoom = Math.min(width / spanX, height / spanY);
        return { centerX, centerY, zoom: Math.max(0.5, Math.min(5, zoom)) };
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    formatNumber(num) {
        return num.toLocaleString();
    },

    // 计算点到线段的距离
    pointToSegmentDistance(px, py, x1, y1, x2, y2) {
        const A = px - x1;
        const B = py - y1;
        const C = x2 - x1;
        const D = y2 - y1;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;

        if (lenSq !== 0) param = dot / lenSq;

        let xx, yy;

        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }

        const dx = px - xx;
        const dy = py - yy;
        return Math.sqrt(dx * dx + dy * dy);
    }
};

// ===== 配置常量 =====
const VEHICLE_COLORS = {
    'Car': '#3B82F6',
    'Suv': '#60A5FA',
    'Truck': '#EF4444',
    'Bus': '#F59E0B',
    'Non_motor_rider': '#10B981',
    'Pedestrian': '#8B5CF6',
    'Unknown': '#9CA3AF'
};

const VEHICLE_SIZES = {
    Car: { length: 4.5, width: 1.8, scale: 1.0 },
    Suv: { length: 4.8, width: 1.9, scale: 1.1 },
    Truck: { length: 12, width: 2.5, scale: 1.5 },
    Bus: { length: 10, width: 2.5, scale: 1.3 },
    Non_motor_rider: { length: 2, width: 0.8, scale: 0.5 },
    Pedestrian: { length: 0.5, width: 0.5, scale: 0.3 },
    Unknown: { length: 4, width: 1.8, scale: 0.8 }
};

// ===== 视图状态 =====
class ViewState {
    constructor() {
        this.width = 800;
        this.height = 600;
        this.centerX = 0;
        this.centerY = 0;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.rotation = 0; // 旋转角度（弧度）
        this.isDragging = false;
        this.isRotating = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.dragButton = 0; // 0: 左键，2: 右键
    }

    setCanvasSize(width, height) {
        this.width = width;
        this.height = height;
    }

    transformPoint(worldX, worldY) {
        // 平移
        let x = worldX - this.centerX;
        let y = worldY - this.centerY;

        // 旋转
        if (this.rotation !== 0) {
            const cos = Math.cos(this.rotation);
            const sin = Math.sin(this.rotation);
            const rotatedX = x * cos - y * sin;
            const rotatedY = x * sin + y * cos;
            x = rotatedX;
            y = rotatedY;
        }

        // 缩放
        x = x * this.zoom;
        y = y * this.zoom;

        // 平移到屏幕中心
        return {
            x: x + this.width / 2 + this.panX,
            y: y + this.height / 2 + this.panY
        };
    }

    inverseTransform(screenX, screenY) {
        // 反向平移
        let x = screenX - this.width / 2 - this.panX;
        let y = screenY - this.height / 2 - this.panY;

        // 反向缩放
        x = x / this.zoom;
        y = y / this.zoom;

        // 反向旋转
        if (this.rotation !== 0) {
            const cos = Math.cos(-this.rotation);
            const sin = Math.sin(-this.rotation);
            const rotatedX = x * cos - y * sin;
            const rotatedY = x * sin + y * cos;
            x = rotatedX;
            y = rotatedY;
        }

        // 反向平移
        return {
            x: x + this.centerX,
            y: y + this.centerY
        };
    }

    startDrag(screenX, screenY, button = 0) {
        this.isDragging = true;
        this.dragButton = button;
        this.lastMouseX = screenX;
        this.lastMouseY = screenY;
    }

    drag(screenX, screenY) {
        if (!this.isDragging) return { dx: 0, dy: 0, dRotation: 0 };

        const dx = screenX - this.lastMouseX;
        const dy = screenY - this.lastMouseY;

        if (this.dragButton === 2) {
            // 右键旋转：根据水平移动计算旋转角度
            const dRotation = dx * 0.005; // 灵敏度
            this.rotation += dRotation;
            this.lastMouseX = screenX;
            this.lastMouseY = screenY;
            return { dx: 0, dy: 0, dRotation };
        } else {
            // 左键平移
            this.panX += dx;
            this.panY += dy;
            this.lastMouseX = screenX;
            this.lastMouseY = screenY;
            return { dx, dy, dRotation: 0 };
        }
    }

    endDrag() {
        this.isDragging = false;
        this.dragButton = 0;
    }

    zoomAt(screenX, screenY, delta) {
        const oldZoom = this.zoom;
        this.zoom = Math.max(0.1, Math.min(10, this.zoom * delta));

        // 以鼠标位置为中心缩放
        // 屏幕坐标转世界坐标：先减去中心点和 pan，再除以旧缩放，再反向旋转
        const cos = Math.cos(-this.rotation);
        const sin = Math.sin(-this.rotation);

        // 鼠标在屏幕空间相对于中心的偏移
        const screenDX = screenX - this.width / 2 - this.panX;
        const screenDY = screenY - this.height / 2 - this.panY;

        // 反向缩放
        const scaledX = screenDX / oldZoom;
        const scaledY = screenDY / oldZoom;

        // 反向旋转得到世界空间偏移
        const worldX = scaledX * cos - scaledY * sin;
        const worldY = scaledX * sin + scaledY * cos;

        // 实际的世界坐标
        const worldPosX = worldX + this.centerX;
        const worldPosY = worldY + this.centerY;

        // 新缩放下，这个世界坐标应该映射到的屏幕位置
        const newScreenX = (worldPosX - this.centerX) * this.zoom + this.width / 2;
        const newScreenY = (worldPosY - this.centerY) * this.zoom + this.height / 2;

        // 调整 pan 使得鼠标位置保持不变
        this.panX += screenX - newScreenX;
        this.panY += screenY - newScreenY;
    }

    reset() {
        this.panX = 0;
        this.panY = 0;
        this.rotation = 0;
        this.zoom = 1;
    }
}

// ===== 地图渲染器 =====
class MapRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.lanes = [];
        this.centerlines = [];
        this.view = new ViewState();
        this.highlightPoint = null;
        this.highlightRadius = 100;
        this.startPoint = null;
        this.endPoint = null;
        this.pathCoords = [];
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.view.setCanvasSize(width, height);
    }

    loadData(mapData) {
        this.lanes = mapData.lanes || [];
        this.centerlines = mapData.centerlines || [];

        // 使用 API 返回的类型信息
        this.lanes.forEach(lane => {
            if (!lane.type) {
                // 如果没有类型，根据颜色推断
                const laneColorTypes = {
                    '#FFD700': 'solid',
                    '#90EE90': 'dashed',
                    '#FF6B6B': 'double_solid',
                    '#87CEEB': 'double_dashed',
                    '#DDA0DD': 'bilateral',
                    '#F0E68C': 'left_dashed_right_solid',
                    '#808080': 'curb',
                    '#8B4513': 'fence',
                    '#00CED1': 'diversion_boundary',
                    '#CCCCCC': 'no_lane',
                };
                lane.type = laneColorTypes[lane.color] || 'unknown';
            }
        });

        // 为中心线添加类型标记
        this.centerlines.forEach(cl => {
            cl.type = 'centerline';
        });

        // 计算边界
        const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
        [...this.lanes, ...this.centerlines].forEach(line => {
            line.x.forEach(x => {
                bounds.minX = Math.min(bounds.minX, x);
                bounds.maxX = Math.max(bounds.maxX, x);
            });
            line.y.forEach(y => {
                bounds.minY = Math.min(bounds.minY, y);
                bounds.maxY = Math.max(bounds.maxY, y);
            });
        });

        // 自动适配视图
        const viewConfig = Utils.fitViewToBounds(bounds, this.view.width, this.view.height);
        this.view.centerX = viewConfig.centerX;
        this.view.centerY = viewConfig.centerY;
        this.view.zoom = viewConfig.zoom;
    }

    setEnabledLaneTypes(types) {
        this.enabledLaneTypes = new Set(types);
    }

    render() {
        const { ctx, view } = this;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 绘制背景
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // 绘制车道线（根据启用的类型过滤）
        const enabledTypes = this.enabledLaneTypes || new Set(['solid', 'dashed', 'double_solid', 'double_dashed', 'bilateral', 'left_dashed_right_solid', 'curb', 'fence', 'diversion_boundary', 'no_lane']);

        this.lanes.forEach(lane => {
            if (enabledTypes.has(lane.type)) {
                this.drawPolyline(lane.x, lane.y, lane.color || '#CCCCCC', 3);
            }
        });

        // 绘制中心线（如果启用）
        if (enabledTypes.has('centerline')) {
            this.centerlines.forEach(cl => {
                this.drawPolyline(cl.x, cl.y, '#4169E1', 2, [5, 5]);
            });
        }

        // 绘制高亮区域
        if (this.highlightPoint) {
            this.drawHighlightCircle(this.highlightPoint, this.highlightRadius);
        }

        // 绘制起点
        if (this.startPoint) {
            this.drawPointMarker(this.startPoint, '#10B981', '起点');
        }

        // 绘制终点
        if (this.endPoint) {
            this.drawPointMarker(this.endPoint, '#EF4444', '终点');
        }

        // 绘制路径
        if (this.pathCoords && this.pathCoords.length > 0) {
            this.drawPath(this.pathCoords);
        }
    }

    drawPolyline(xCoords, yCoords, color, width, dashPattern = null) {
        const { ctx, view } = this;
        if (xCoords.length < 2) return;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (dashPattern) {
            ctx.setLineDash(dashPattern);
        } else {
            ctx.setLineDash([]);
        }

        const start = view.transformPoint(xCoords[0], yCoords[0]);
        ctx.moveTo(start.x, start.y);

        for (let i = 1; i < xCoords.length; i++) {
            const pt = view.transformPoint(xCoords[i], yCoords[i]);
            ctx.lineTo(pt.x, pt.y);
        }

        ctx.stroke();
        ctx.setLineDash([]);
    }

    drawHighlightCircle(point, radius) {
        const { ctx, view } = this;
        const pos = view.transformPoint(point[0], point[1]);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * view.zoom, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        ctx.fill();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2;
        ctx.stroke();

        // 中心点
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#10B981';
        ctx.fill();
    }

    drawPointMarker(point, color, label) {
        const { ctx, view } = this;
        const pos = view.transformPoint(point[0], point[1]);

        // 星形标记
        ctx.beginPath();
        ctx.fillStyle = color;
        const spikes = 5;
        const outerRadius = 12;
        const innerRadius = 6;
        let rot = Math.PI / 2 * 3;
        let x = pos.x;
        let y = pos.y;
        const step = Math.PI / spikes;

        for (let i = 0; i < spikes; i++) {
            x = pos.x + Math.cos(rot) * outerRadius;
            y = pos.y + Math.sin(rot) * outerRadius;
            ctx.lineTo(x, y);
            rot += step;

            x = pos.x + Math.cos(rot) * innerRadius;
            y = pos.y + Math.sin(rot) * innerRadius;
            ctx.lineTo(x, y);
            rot += step;
        }

        ctx.lineTo(pos.x, pos.y - outerRadius);
        ctx.closePath();
        ctx.fill();

        // 标签
        if (label) {
            ctx.fillStyle = color;
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(label, pos.x, pos.y - 20);
        }
    }

    drawPath(coords) {
        if (coords.length < 2) return;
        const { ctx, view } = this;

        ctx.beginPath();
        ctx.strokeStyle = '#1E90FF';
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const start = view.transformPoint(coords[0][0], coords[0][1]);
        ctx.moveTo(start.x, start.y);

        for (let i = 1; i < coords.length; i++) {
            const pt = view.transformPoint(coords[i][0], coords[i][1]);
            ctx.lineTo(pt.x, pt.y);
        }

        ctx.stroke();
    }

    setHighlightPoint(point) {
        this.highlightPoint = point;
    }

    setStartPoint(point) {
        this.startPoint = point;
    }

    setEndPoint(point) {
        this.endPoint = point;
    }

    setPathCoords(coords) {
        this.pathCoords = coords;
    }

    clearMarkers() {
        this.startPoint = null;
        this.endPoint = null;
        this.pathCoords = [];
    }
}

// ===== 轨迹渲染器 =====
class TrajectoryRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.view = new ViewState();
        this.trajectories = [];
        this.trailLength = 30;
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.view.setCanvasSize(width, height);
    }

    setTrajectories(trajectories) {
        this.trajectories = trajectories;
    }

    render(currentFrameIdx, showTrail = true) {
        const { ctx, view } = this;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!showTrail) return;

        // 绘制轨迹拖尾
        this.trajectories.forEach(traj => {
            const states = traj.states || [];
            if (states.length < 2) return;

            const color = VEHICLE_COLORS[traj.vehicle_type] || VEHICLE_COLORS.Unknown;

            // 只显示当前帧之前的轨迹点
            const visibleStates = states.filter(s => s.frame_id <= currentFrameIdx);
            if (visibleStates.length < 2) return;

            // 限制拖尾长度
            const trailStates = visibleStates.slice(-this.trailLength);

            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([3, 3]);
            ctx.globalAlpha = 0.5;

            const start = view.transformPoint(
                trailStates[0].position[0],
                trailStates[0].position[1]
            );
            ctx.moveTo(start.x, start.y);

            for (let i = 1; i < trailStates.length; i++) {
                const pt = view.transformPoint(
                    trailStates[i].position[0],
                    trailStates[i].position[1]
                );
                ctx.lineTo(pt.x, pt.y);
            }

            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
        });
    }
}

// ===== 车辆渲染器 =====
class VehicleRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.view = new ViewState();
        this.vehicles = [];
        this.hoveredVehicle = null;
        this.onVehicleHover = null;
        // 车辆尺寸基于实际大小，考虑缩放
        this.baseVehicleScale = 0.8; // 缩小车辆显示比例
        // 启用的目标类型
        this.enabledObjectTypes = new Set(['Car', 'Suv', 'Truck', 'Bus', 'Non_motor_rider', 'Pedestrian', 'Unknown']);
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.view.setCanvasSize(width, height);
    }

    setVehicles(vehicles) {
        this.vehicles = vehicles;
    }

    setEnabledObjectTypes(types) {
        this.enabledObjectTypes = new Set(types);
    }

    render(showIds = true) {
        const { ctx, view } = this;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.vehicles.forEach(vehicle => {
            // 检查是否启用了该目标类型
            if (!this.enabledObjectTypes.has(vehicle.vehicle_type)) {
                return; // 跳过未启用的类型
            }

            const pos = view.transformPoint(
                vehicle.position[0],
                vehicle.position[1]
            );

            const color = VEHICLE_COLORS[vehicle.vehicle_type] || VEHICLE_COLORS.Unknown;
            const vehicleConfig = VEHICLE_SIZES[vehicle.vehicle_type] || VEHICLE_SIZES.Unknown;
            const isHovered = this.hoveredVehicle === vehicle;

            // 计算显示尺寸：基于实际尺寸和缩放比例
            // 地图坐标系到屏幕坐标系的转换
            const baseSize = 4; // 基础像素大小
            const screenScale = this.baseVehicleScale * vehicleConfig.scale;
            const size = Math.max(3, baseSize * screenScale * view.zoom * 0.3);

            // 绘制车辆（椭圆形，带方向）
            this.drawVehicle(ctx, pos.x, pos.y, size, color, vehicle.heading || 0, isHovered);

            // 绘制 ID
            if (showIds) {
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 9px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;
                ctx.strokeText(String(vehicle.vehicle_id), pos.x, pos.y - size - 4);
                ctx.fillText(String(vehicle.vehicle_id), pos.x, pos.y - size - 4);
            }
        });
    }

    drawVehicle(ctx, x, y, size, color, heading, isHovered) {
        ctx.save();

        // 旋转到车辆朝向
        ctx.translate(x, y);
        ctx.rotate(-heading * Math.PI / 180);

        // 绘制车身（椭圆）
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.ellipse(0, 0, size * 1.5, size, 0, 0, Math.PI * 2);
        ctx.fill();

        // 绘制边框
        ctx.strokeStyle = 'white';
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();

        // 绘制车头指示
        ctx.beginPath();
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.moveTo(size * 0.5, 0);
        ctx.lineTo(-size * 0.3, -size * 0.5);
        ctx.lineTo(-size * 0.3, size * 0.5);
        ctx.closePath();
        ctx.fill();

        ctx.restore();
    }

    getVehicleAtPoint(screenX, screenY) {
        const hitRadius = 15;
        for (let i = this.vehicles.length - 1; i >= 0; i--) {
            const vehicle = this.vehicles[i];
            const pos = this.view.transformPoint(
                vehicle.position[0],
                vehicle.position[1]
            );
            const dist = Math.sqrt(
                (screenX - pos.x) ** 2 + (screenY - pos.y) ** 2
            );
            if (dist <= hitRadius) {
                return vehicle;
            }
        }
        return null;
    }
}

// ===== 主可视化器 =====
class TrafficFlowVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // 初始化 Canvas
        this.mapCanvas = document.getElementById('map-canvas');
        this.trajCanvas = document.getElementById('trajectory-canvas');
        this.vehicleCanvas = document.getElementById('vehicle-canvas');

        // 初始化渲染器
        this.mapRenderer = new MapRenderer(this.mapCanvas);
        this.trajRenderer = new TrajectoryRenderer(this.trajCanvas);
        this.vehicleRenderer = new VehicleRenderer(this.vehicleCanvas);

        // 状态
        this.frames = [];
        this.trajectories = [];
        this.currentFrameIdx = 0;
        this.isPlaying = false;
        this.fps = 15;
        this.playTimer = null;

        // 选点模式
        this.selectMode = ''; // '', 'start', 'end'

        // 显示选项
        this.options = {
            showMap: true,
            showVehicles: true,
            showIds: true,
            showTrajectories: false,
            showTrail: true
        };

        // 绑定事件
        this.bindEvents();

        // 初始化尺寸
        this.resize();
    }

    resize() {
        const rect = this.container.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        this.mapRenderer.resize(width, height);
        this.trajRenderer.resize(width, height);
        this.vehicleRenderer.resize(width, height);

        // 同步视图
        const mainView = this.mapRenderer.view;
        this.trajRenderer.view = new ViewState();
        Object.assign(this.trajRenderer.view, {
            width, height,
            centerX: mainView.centerX,
            centerY: mainView.centerY,
            zoom: mainView.zoom,
            panX: mainView.panX,
            panY: mainView.panY
        });

        this.vehicleRenderer.view = new ViewState();
        Object.assign(this.vehicleRenderer.view, {
            width, height,
            centerX: mainView.centerX,
            centerY: mainView.centerY,
            zoom: mainView.zoom,
            panX: mainView.panX,
            panY: mainView.panY
        });

        this.render();
    }

    bindEvents() {
        // ===== 鼠标交互 =====

        // 阻止浏览器默认右键菜单 - 绑定到容器
        this.container.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            return false;
        });

        // 地图拖动（左键平移，右键旋转） - 绑定到容器
        this.container.addEventListener('mousedown', (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            if (this.selectMode) {
                // 选点模式 - 仅左键有效
                if (e.button === 0) {
                    const worldPos = this.mapRenderer.view.inverseTransform(x, y);
                    this.handlePointSelect(worldPos);
                }
            } else {
                // 拖动模式 - 左键平移，右键旋转
                const button = e.button; // 0: 左键，2: 右键
                this.mapRenderer.view.startDrag(x, y, button);
                this.trajRenderer.view.startDrag(x, y, button);
                this.vehicleRenderer.view.startDrag(x, y, button);
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.mapRenderer.view.isDragging) return;

            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // 只拖动地图视图，然后同步到其他视图
            this.mapRenderer.view.drag(x, y);

            // 同步轨迹和车辆视图到地图视图
            const mapView = this.mapRenderer.view;
            [this.trajRenderer.view, this.vehicleRenderer.view].forEach(view => {
                view.panX = mapView.panX;
                view.panY = mapView.panY;
                view.rotation = mapView.rotation;
            });

            this.render();
        });

        window.addEventListener('mouseup', () => {
            this.mapRenderer.view.endDrag();
            this.trajRenderer.view.endDrag();
            this.vehicleRenderer.view.endDrag();
        });

        // 滚轮缩放 - 绑定到容器而不是单个 canvas，因为 canvas 有层叠
        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();

            // 使用容器的坐标计算
            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // 计算缩放因子
            const delta = e.deltaY > 0 ? 0.9 : 1.1;

            // 在所有视图上应用缩放
            this.mapRenderer.view.zoomAt(x, y, delta);

            // 同步轨迹和车辆视图到地图视图（完全同步）
            const mapView = this.mapRenderer.view;
            [this.trajRenderer.view, this.vehicleRenderer.view].forEach(view => {
                view.zoom = mapView.zoom;
                view.panX = mapView.panX;
                view.panY = mapView.panY;
                view.centerX = mapView.centerX;
                view.centerY = mapView.centerY;
                view.rotation = mapView.rotation;
            });

            // 更新缩放滑块
            document.getElementById('zoom-slider').value = this.mapRenderer.view.zoom;
            document.getElementById('zoom-value').textContent = this.mapRenderer.view.zoom.toFixed(1);

            this.render();
        }, { passive: false });

        // 双击重置视图
        this.mapCanvas.addEventListener('dblclick', () => {
            this.resetView();
        });

        // 窗口大小变化
        window.addEventListener('resize', Utils.debounce(() => this.resize(), 100));
    }

    handlePointSelect(worldPos) {
        const point = [worldPos.x, worldPos.y, 0];

        if (this.selectMode === 'start') {
            this.mapRenderer.setStartPoint(point);
            this.triggerEvent('pointSelected', { mode: 'start', point });
        } else if (this.selectMode === 'end') {
            this.mapRenderer.setEndPoint(point);
            this.triggerEvent('pointSelected', { mode: 'end', point });
        }

        this.selectMode = '';
        document.body.classList.remove('mode-start', 'mode-end');
        this.render();
    }

    triggerEvent(eventName, data) {
        const event = new CustomEvent(eventName, { detail: data });
        window.dispatchEvent(event);
    }

    loadMapData(mapData) {
        this.mapRenderer.loadData(mapData);

        // 同步视图到其他渲染器
        const mainView = this.mapRenderer.view;
        Object.assign(this.trajRenderer.view, {
            centerX: mainView.centerX,
            centerY: mainView.centerY,
            zoom: mainView.zoom,
            panX: mainView.panX,
            panY: mainView.panY
        });
        Object.assign(this.vehicleRenderer.view, {
            centerX: mainView.centerX,
            centerY: mainView.centerY,
            zoom: mainView.zoom,
            panX: mainView.panX,
            panY: mainView.panY
        });

        this.render();
    }

    loadTrafficData(frames, trajectories) {
        this.frames = frames;
        this.trajectories = trajectories;
        this.currentFrameIdx = 0;

        // 更新 UI
        document.getElementById('frame-count').textContent = frames.length;
        document.getElementById('total-frames').textContent = frames.length;
        document.getElementById('frame-slider').max = frames.length - 1;
        document.getElementById('current-frame-label').textContent = '0';

        // 设置第一帧的车辆
        if (frames.length > 0 && frames[0].vehicles) {
            this.vehicleRenderer.setVehicles(frames[0].vehicles);
            document.getElementById('vehicle-count').textContent = frames[0].vehicles.length;
        }

        this.render();
    }

    setFrame(idx) {
        if (idx < 0 || idx >= this.frames.length) return;

        this.currentFrameIdx = idx;
        document.getElementById('current-frame').textContent = idx;
        document.getElementById('current-frame-label').textContent = idx;
        document.getElementById('frame-slider').value = idx;

        const frame = this.frames[idx];
        document.getElementById('vehicle-count').textContent = frame.vehicles?.length || 0;

        this.vehicleRenderer.setVehicles(frame.vehicles || []);
        this.render();
    }

    play() {
        if (this.isPlaying) return;

        this.isPlaying = true;
        document.getElementById('btn-play').textContent = '⏸️';
        document.getElementById('btn-play').classList.add('playing');

        const interval = 1000 / this.fps;
        this.playTimer = setInterval(() => {
            let nextIdx = this.currentFrameIdx + 1;
            if (nextIdx >= this.frames.length) {
                nextIdx = 0;
            }
            this.setFrame(nextIdx);
        }, interval);
    }

    pause() {
        this.isPlaying = false;
        document.getElementById('btn-play').textContent = '▶️';
        document.getElementById('btn-play').classList.remove('playing');

        if (this.playTimer) {
            clearInterval(this.playTimer);
            this.playTimer = null;
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    setFps(fps) {
        this.fps = fps;
        document.getElementById('fps-value').textContent = fps;

        if (this.isPlaying) {
            this.pause();
            this.play();
        }
    }

    setZoom(zoom) {
        const view = this.mapRenderer.view;
        view.zoom = Math.max(0.5, Math.min(5, zoom));

        // 同步到其他渲染器（完全同步）
        [this.trajRenderer.view, this.vehicleRenderer.view].forEach(v => {
            v.zoom = view.zoom;
            v.panX = view.panX;
            v.panY = view.panY;
            v.centerX = view.centerX;
            v.centerY = view.centerY;
            v.rotation = view.rotation;
        });

        document.getElementById('zoom-value').textContent = view.zoom.toFixed(1);
        document.getElementById('zoom-slider').value = view.zoom;

        this.render();
    }

    pan(dx, dy) {
        const view = this.mapRenderer.view;
        view.panX += dx;
        view.panY += dy;

        // 同步到其他渲染器（完全同步）
        [this.trajRenderer.view, this.vehicleRenderer.view].forEach(v => {
            v.panX = view.panX;
            v.panY = view.panY;
        });

        this.render();
    }

    resetView() {
        const view = this.mapRenderer.view;
        view.reset();

        // 同步到其他渲染器
        this.trajRenderer.view.zoom = view.zoom;
        this.trajRenderer.view.panX = view.panX;
        this.trajRenderer.view.panY = view.panY;
        this.trajRenderer.view.rotation = view.rotation;
        this.vehicleRenderer.view.zoom = view.zoom;
        this.vehicleRenderer.view.panX = view.panX;
        this.vehicleRenderer.view.panY = view.panY;
        this.vehicleRenderer.view.rotation = view.rotation;

        // 更新 UI
        document.getElementById('zoom-slider').value = view.zoom;
        document.getElementById('zoom-value').textContent = view.zoom.toFixed(1);

        this.render();
    }

    setSelectMode(mode) {
        this.selectMode = mode;
        document.body.classList.remove('mode-start', 'mode-end');
        if (mode === 'start') {
            document.body.classList.add('mode-start');
        } else if (mode === 'end') {
            document.body.classList.add('mode-end');
        }
    }

    setOption(key, value) {
        this.options[key] = value;
        this.render();
    }

    render() {
        if (this.options.showMap) {
            this.mapRenderer.render();
        } else {
            this.mapRenderer.ctx.clearRect(0, 0, this.mapCanvas.width, this.mapCanvas.height);
        }

        if (this.options.showTrajectories || this.options.showTrail) {
            this.trajRenderer.render(this.currentFrameIdx, this.options.showTrail);
        } else {
            this.trajRenderer.ctx.clearRect(0, 0, this.trajCanvas.width, this.trajCanvas.height);
        }

        if (this.options.showVehicles) {
            this.vehicleRenderer.render(this.options.showIds);
        } else {
            this.vehicleRenderer.ctx.clearRect(0, 0, this.vehicleCanvas.width, this.vehicleCanvas.height);
        }
    }

    updateStats(stats) {
        const container = document.getElementById('stats-container');
        if (!container) return; // 元素不存在，直接返回

        if (!stats) {
            container.innerHTML = '<p>请先加载数据</p>';
            return;
        }

        let html = '';
        for (const [key, value] of Object.entries(stats)) {
            html += `<div class="stat-row">
                <span>${key}:</span>
                <span>${typeof value === 'number' ? Utils.formatNumber(value) : value}</span>
            </div>`;
        }
        container.innerHTML = html;
    }
}

// ===== 应用程序入口 =====
let visualizer = null;

document.addEventListener('DOMContentLoaded', () => {
    visualizer = new TrafficFlowVisualizer('canvas-container');
    bindControlEvents();
    initTabs();
});

function initLLMProgressStream() {
    // 连接 SSE 流
    const eventSource = new EventSource('/api/llm_progress');

    let lastThinkingEvent = null;

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        showLLMReasoningInChat(data.event_type, data.data);
    };

    eventSource.onerror = function() {
        // 连接错误，静默重试
        eventSource.close();
    };
}

function showLLMReasoningInChat(eventType, data) {
    const historyEl = document.getElementById('chat-history');

    // 构建消息内容
    let title, detail, className;

    switch(eventType) {
        case 'lane_analysis_start':
            title = `🔍 车道分析 - ${data.lane_id}`;
            detail = `车辆数变化：${data.prev_count} → ${data.curr_count} (${data.diff >= 0 ? '+' : ''}${data.diff})`;
            className = 'start';
            break;
        case 'llm_thinking':
            title = `🧠 LLM 思考中...`;
            if (data.analysis_type === 'id_consistency') {
                detail = `分析轨迹 #${data.track_id} 的 ID 一致性`;
            } else {
                detail = '分析车道数量守恒和遮挡情况';
            }
            className = 'thinking';
            break;
        case 'lane_analysis_result':
            title = `✅ 车道分析完成`;
            detail = data.result?.reasoning || `原因：${data.result?.cause || 'unknown'}`;
            className = 'result';
            break;
        case 'occlusion_analysis_start':
            title = `🔍 遮挡分析 - 轨迹 #${data.track_id}`;
            detail = `丢失帧数：${data.lost_frames}, 附近车辆：${data.nearby_count}`;
            className = 'start';
            break;
        case 'occlusion_analysis_result':
            title = `✅ 遮挡分析完成`;
            detail = data.result?.reasoning || (data.result?.is_occluded ? '检测到遮挡' : '未检测到遮挡');
            className = 'result';
            break;
        case 'id_analysis_start':
            title = `🔍 ID 一致性分析 - 轨迹 #${data.track_id}`;
            detail = `帧 ${data.frame_id}, 历史帧数：${data.history_length}`;
            className = 'start';
            break;
        case 'id_analysis_result':
            title = `✅ ID 分析完成`;
            const decisionMap = {
                'keep_id': '应保持原 ID',
                'new_target': '新目标',
                'merge_ids': '合并 ID'
            };
            detail = data.result?.reasoning || decisionMap[data.result?.decision] || '分析完成';
            className = 'result';
            break;
        case 'lane_analysis_error':
        case 'occlusion_analysis_error':
        case 'id_analysis_error':
            title = `❌ 分析失败`;
            detail = data.error || data.message || '未知错误';
            // 如果是详细错误（包含 traceback），只显示第一行
            if (detail && detail.includes('\\n')) {
                detail = detail.split('\\n')[0];
            }
            className = 'error';
            break;
        default:
            title = eventType;
            detail = JSON.stringify(data);
            className = '';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message llm-reasoning ${className}`;
    messageDiv.innerHTML = `
        <div class="llm-reasoning-title">${title}</div>
        <div class="llm-reasoning-detail">${escapeHtml(detail)}</div>
    `;

    historyEl.appendChild(messageDiv);
    historyEl.scrollTop = historyEl.scrollHeight;
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // 移除所有 active 类
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // 添加 active 类
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function bindControlEvents() {
    // 加载地图按钮
    document.getElementById('load-map-btn').addEventListener('click', async () => {
        const statusEl = document.getElementById('load-status');
        const loadingEl = document.getElementById('loading-overlay');

        loadingEl.classList.remove('hidden');
        statusEl.textContent = '';
        statusEl.className = 'status-message';

        try {
            const response = await fetch('/api/map_data');
            const mapData = await response.json();

            visualizer.loadMapData(mapData);
            statusEl.textContent = `地图加载成功！车道数：${mapData.lanes.length}`;
            statusEl.className = 'status-message success';
        } catch (error) {
            statusEl.textContent = `错误：${error.message}`;
            statusEl.className = 'status-message error';
        } finally {
            loadingEl.classList.add('hidden');
        }
    });

    // 交通流重建按钮
    document.getElementById('load-traffic-btn').addEventListener('click', async () => {
        const path = document.getElementById('detection-path').value;
        const startFrame = parseInt(document.getElementById('start-frame').value);
        const endFrame = parseInt(document.getElementById('end-frame').value);
        const useLlm = document.getElementById('use-llm-optimize').checked;

        // 获取 LLM 配置（如果启用 LLM）
        let llmProvider = 'deepseek';
        let llmApiKey = '';
        let llmPort = 8000;

        if (useLlm) {
            llmProvider = document.getElementById('llm-provider').value;
            llmApiKey = document.getElementById('llm-api-key').value;
            llmPort = parseInt(document.getElementById('llm-port').value) || 8000;
        }

        const statusEl = document.getElementById('load-status');
        const loadingEl = document.getElementById('loading-overlay');

        loadingEl.classList.remove('hidden');
        statusEl.textContent = '';
        statusEl.className = 'status-message';

        // 调试输出
        console.log('[DEBUG] 交通流重建请求:', {
            path,
            start_frame: startFrame,
            end_frame: endFrame,
            use_llm: useLlm,
            llm_provider: llmProvider,
            has_api_key: !!llmApiKey
        });

        // 在对话窗口显示开始消息
        const historyEl = document.getElementById('chat-history');
        if (useLlm) {
            historyEl.innerHTML += `
                <div class="chat-message llm-reasoning start">
                    <div class="llm-reasoning-title">🚀 开始交通流重建（LLM 增强模式）</div>
                    <div class="llm-reasoning-detail">
                        提供商：${llmProvider} |
                        每帧分析每条轨迹的 ID 一致性
                    </div>
                </div>`;
            historyEl.scrollTop = historyEl.scrollHeight;
        } else {
            historyEl.innerHTML += `
                <div class="chat-message system">
                    开始交通流重建（纯 DeepSORT 模式，未启用 LLM）
                </div>`;
            historyEl.scrollTop = historyEl.scrollHeight;
        }

        try {
            const response = await fetch('/api/reconstruct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    path,
                    start_frame: startFrame,
                    end_frame: endFrame,
                    use_llm: useLlm,
                    llm_provider: llmProvider,
                    llm_api_key: llmApiKey,
                    llm_port: llmPort
                })
            });

            const result = await response.json();

            if (result.success) {
                visualizer.loadTrafficData(result.frames, result.trajectories);
                visualizer.updateStats(result.statistics);

                const modeText = useLlm ? '（LLM 增强模式）' : '';
                statusEl.textContent = `重建成功${modeText}！共 ${result.frames.length} 帧，${result.total_vehicles} 辆车`;
                statusEl.className = 'status-message success';

                // 显示完成消息
                if (useLlm) {
                    historyEl.innerHTML += `
                        <div class="chat-message llm-reasoning result">
                            <div class="llm-reasoning-title">✅ 交通流重建完成</div>
                            <div class="llm-reasoning-detail">共 ${result.total_vehicles} 辆车，LLM 调用 ${result.statistics?.llm_calls || 0} 次</div>
                        </div>`;
                    historyEl.scrollTop = historyEl.scrollHeight;
                }
            } else {
                throw new Error(result.error || '重建失败');
            }
        } catch (error) {
            statusEl.textContent = `错误：${error.message}`;
            statusEl.className = 'status-message error';

            if (useLlm) {
                historyEl.innerHTML += `
                    <div class="chat-message llm-reasoning error">
                        <div class="llm-reasoning-title">❌ 重建失败</div>
                        <div class="llm-reasoning-detail">${escapeHtml(error.message)}</div>
                    </div>`;
                historyEl.scrollTop = historyEl.scrollHeight;
            }
        } finally {
            loadingEl.classList.add('hidden');
        }
    });

    // 播放控制
    document.getElementById('btn-play').addEventListener('click', () => {
        visualizer.togglePlay();
    });

    document.getElementById('btn-first').addEventListener('click', () => {
        visualizer.pause();
        visualizer.setFrame(0);
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
        visualizer.pause();
        visualizer.setFrame(visualizer.currentFrameIdx - 1);
    });

    document.getElementById('btn-next').addEventListener('click', () => {
        visualizer.pause();
        visualizer.setFrame(visualizer.currentFrameIdx + 1);
    });

    document.getElementById('btn-last').addEventListener('click', () => {
        visualizer.pause();
        visualizer.setFrame(visualizer.frames.length - 1);
    });

    // 帧滑块
    document.getElementById('frame-slider').addEventListener('input', (e) => {
        const idx = parseInt(e.target.value);
        visualizer.pause();
        visualizer.setFrame(idx);
    });

    // FPS 控制
    document.getElementById('fps-slider').addEventListener('input', (e) => {
        visualizer.setFps(parseInt(e.target.value));
    });

    // 显示选项
    document.getElementById('show-map').addEventListener('change', (e) => {
        visualizer.setOption('showMap', e.target.checked);
    });

    document.getElementById('show-vehicles').addEventListener('change', (e) => {
        visualizer.setOption('showVehicles', e.target.checked);
    });

    document.getElementById('show-ids').addEventListener('change', (e) => {
        visualizer.setOption('showIds', e.target.checked);
    });

    document.getElementById('show-trajectories').addEventListener('change', (e) => {
        visualizer.setOption('showTrajectories', e.target.checked);
    });

    document.getElementById('show-trail').addEventListener('change', (e) => {
        visualizer.setOption('showTrail', e.target.checked);
    });

    // 视图控制
    document.getElementById('zoom-slider').addEventListener('input', (e) => {
        visualizer.setZoom(parseFloat(e.target.value));
    });

    document.getElementById('btn-zoom-in').addEventListener('click', () => {
        visualizer.setZoom(visualizer.mapRenderer.view.zoom * 1.2);
    });

    document.getElementById('btn-zoom-out').addEventListener('click', () => {
        visualizer.setZoom(visualizer.mapRenderer.view.zoom / 1.2);
    });

    document.getElementById('btn-reset-view').addEventListener('click', () => {
        visualizer.resetView();
    });

    // 选点按钮
    document.getElementById('set-start-btn').addEventListener('click', () => {
        visualizer.setSelectMode('start');
    });

    document.getElementById('set-end-btn').addEventListener('click', () => {
        visualizer.setSelectMode('end');
    });

    document.getElementById('clear-positions-btn').addEventListener('click', () => {
        visualizer.mapRenderer.clearMarkers();
        visualizer.render();
    });

    // 键盘快捷键
    document.addEventListener('keydown', (e) => {
        switch(e.code) {
            case 'Space':
                e.preventDefault();
                visualizer.togglePlay();
                break;
            case 'ArrowLeft':
                visualizer.setFrame(visualizer.currentFrameIdx - 1);
                break;
            case 'ArrowRight':
                visualizer.setFrame(visualizer.currentFrameIdx + 1);
                break;
            case 'Home':
                visualizer.setFrame(0);
                break;
            case 'End':
                visualizer.setFrame(visualizer.frames.length - 1);
                break;
            case 'Escape':
                visualizer.setSelectMode('');
                break;
        }
    });

    // 监听选点事件
    window.addEventListener('pointSelected', (e) => {
        const { mode, point } = e.detail;

        if (mode === 'start') {
            document.getElementById('start-x').value = point[0].toFixed(2);
            document.getElementById('start-y').value = point[1].toFixed(2);
            showPathStatus(`起点已设置：(${point[0].toFixed(1)}, ${point[1].toFixed(1)})`, 'success');
        } else if (mode === 'end') {
            document.getElementById('end-x').value = point[0].toFixed(2);
            document.getElementById('end-y').value = point[1].toFixed(2);
            showPathStatus(`终点已设置：(${point[0].toFixed(1)}, ${point[1].toFixed(1)})`, 'success');
        }
    });

    // 路径规划
    document.getElementById('find-path-btn').addEventListener('click', async () => {
        const originX = parseFloat(document.getElementById('start-x').value);
        const originY = parseFloat(document.getElementById('start-y').value);
        const destX = parseFloat(document.getElementById('end-x').value);
        const destY = parseFloat(document.getElementById('end-y').value);

        if (isNaN(originX) || isNaN(originY) || isNaN(destX) || isNaN(destY)) {
            showPathStatus('请先设置起点和终点', 'error');
            return;
        }

        try {
            const response = await fetch('/api/find_path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    origin: [originX, originY, 0],
                    destination: [destX, destY, 0]
                })
            });

            const result = await response.json();
            if (result.success) {
                const path = result.path;
                visualizer.mapRenderer.setPathCoords(path.waypoints);
                visualizer.render();
                showPathStatus(`路径规划成功！长度：${path.length.toFixed(1)}m`, 'success');
            } else {
                showPathStatus(`路径规划失败：${result.error}`, 'error');
            }
        } catch (error) {
            showPathStatus(`请求失败：${error.message}`, 'error');
        }
    });

    // 清空位置
    document.getElementById('clear-positions-btn').addEventListener('click', () => {
        visualizer.mapRenderer.clearMarkers();
        visualizer.render();
        document.getElementById('start-x').value = '';
        document.getElementById('start-y').value = '';
        document.getElementById('end-x').value = '';
        document.getElementById('end-y').value = '';
        showPathStatus('已清空起点和终点', 'success');
    });

    // LLM 提供商切换
    document.getElementById('llm-provider').addEventListener('change', (e) => {
        const provider = e.target.value;
        const portGroup = document.getElementById('local-port-group');
        if (provider === 'qwen' || provider === 'gemma4') {
            portGroup.style.display = 'block';
        } else {
            portGroup.style.display = 'none';
        }
    });

    // 发送聊天消息
    document.getElementById('send-chat-btn').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // 清空聊天历史
    document.getElementById('clear-chat-btn').addEventListener('click', () => {
        const historyEl = document.getElementById('chat-history');
        historyEl.innerHTML = '<div class="chat-message system">系统已就绪，请输入问题...</div>';
    });

    // 初始化 LLM 推理过程 SSE 连接
    initLLMProgressStream();

    // 车道图层控制
    document.querySelectorAll('.lane-type').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const enabledTypes = Array.from(document.querySelectorAll('.lane-type:checked'))
                .map(cb => cb.value);

            // 过滤显示的图层
            visualizer.mapRenderer.setEnabledLaneTypes(enabledTypes);
            visualizer.render();
        });
    });

    // 目标类型过滤控制
    document.querySelectorAll('.object-type').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const enabledTypes = Array.from(document.querySelectorAll('.object-type:checked'))
                .map(cb => cb.value);

            // 设置车辆渲染器的目标类型过滤
            visualizer.vehicleRenderer.setEnabledObjectTypes(enabledTypes);
            visualizer.render();
        });
    });

    // 初始化默认启用的图层类型
    const defaultEnabledTypes = Array.from(document.querySelectorAll('.lane-type:checked'))
        .map(cb => cb.value);
    if (visualizer && visualizer.mapRenderer) {
        visualizer.mapRenderer.setEnabledLaneTypes(defaultEnabledTypes);
    }

    // 初始化默认启用的目标类型
    const defaultEnabledObjectTypes = Array.from(document.querySelectorAll('.object-type:checked'))
        .map(cb => cb.value);
    if (visualizer && visualizer.vehicleRenderer) {
        visualizer.vehicleRenderer.setEnabledObjectTypes(defaultEnabledObjectTypes);
    }
}

function showPathStatus(message, type = 'success') {
    const statusEl = document.getElementById('path-status');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;
}

async function sendChatMessage() {
    const inputEl = document.getElementById('chat-input');
    const message = inputEl.value.trim();
    if (!message) return;

    // 添加用户消息到历史
    const historyEl = document.getElementById('chat-history');
    historyEl.innerHTML += `<div class="chat-message user">${escapeHtml(message)}</div>`;
    inputEl.value = '';
    historyEl.scrollTop = historyEl.scrollHeight;

    // 获取 LLM 配置
    const provider = document.getElementById('llm-provider').value;
    const apiKey = document.getElementById('llm-api-key').value;
    const port = document.getElementById('llm-port').value;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                provider,
                api_key: apiKey,
                port: parseInt(port),
                context: {
                    start_position: visualizer.mapRenderer.startPoint,
                    end_position: visualizer.mapRenderer.endPoint
                }
            })
        });

        const result = await response.json();
        if (result.success) {
            historyEl.innerHTML += `<div class="chat-message assistant">${escapeHtml(result.response)}</div>`;

            // 如果返回了路径，更新显示
            if (result.path) {
                visualizer.mapRenderer.setPathCoords(result.path.waypoints);
                visualizer.render();
            }
        } else {
            historyEl.innerHTML += `<div class="chat-message assistant error">错误：${result.error}</div>`;
        }
    } catch (error) {
        historyEl.innerHTML += `<div class="chat-message assistant error">请求失败：${error.message}</div>`;
    }

    historyEl.scrollTop = historyEl.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 导出到全局
window.trafficFlowVisualizer = visualizer;
