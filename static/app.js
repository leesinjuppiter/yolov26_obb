'use strict';

const byId = id => document.getElementById(id);
const statuses = {pending: '待审核', in_progress: '审核中', done: '已完成', negative: '确认无目标', uncertain: '无法判断'};
const eventKinds = {one2one_only: 'one2one 独有框', one2many_only: 'one2many 独有框', localization: '定位分歧', class_conflict: '类别冲突'};
const decisions = ['预测正确', 'one2one 漏检', 'one2many 漏检', 'one2one 误检', 'one2many 误检',
  'one2one 错分', 'one2many 错分', 'one2one 定位偏差', 'one2many 定位偏差',
  'one2one 重复框', 'one2many 重复框', '原标注有误', '无法判断'];
const color = {label: '#80e8ca', original: '#f4ec91', one2one: '#63b8ff', one2many: '#ff9a79', selected: '#ffffff'};
const canvas = byId('canvas');
const context = canvas.getContext('2d');
const layerIds = ['layer-labels', 'layer-original', 'layer-one2one', 'layer-one2many'];
let task, frame, image, selected, highlighted, gesture, drawing, mode = 'select', dirty = false, busy = false;
let originalBoxes = [];
const hiddenBoxesByFrame = new Map();
let hiddenBoxes = new Set();

function make(tag, content, className) {
  const element = document.createElement(tag);
  if (content != null) element.textContent = content;
  if (className) element.className = className;
  return element;
}

function button(text, callback) {
  const element = make('button', text);
  element.type = 'button';
  element.addEventListener('click', callback);
  return element;
}

function field(label, element) {
  const wrapper = make('label', label);
  wrapper.append(element);
  return wrapper;
}

function message(text, error = false) {
  byId('message').textContent = text;
  byId('message').dataset.error = String(error);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      detail = parsed.error || parsed.detail || parsed.message || text;
    } catch { /* Plain HTTP error */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.status === 204 ? null : response.json();
}

function zipOptions(file) {
  return {method: 'POST', headers: {'Content-Type': 'application/zip'}, body: file};
}

function taskPath(id = task.id) {
  return '/api/tasks/' + encodeURIComponent(id);
}

function framePath(id = frame.id) {
  return taskPath() + '/images/' + encodeURIComponent(id);
}

async function run(action) {
  if (busy) return;
  busy = true;
  const review = document.querySelector('.review');
  const editor = document.querySelector('.editor');
  review.inert = editor.inert = true;
  try {
    await action();
  } catch (error) {
    message(error.message || '操作失败', true);
  } finally {
    busy = false;
    review.inert = editor.inert = false;
  }
}

function markDirty() {
  dirty = true;
  message('当前图片有尚未保存的修改');
}

function canLeave() {
  return !dirty || window.confirm('当前图片有尚未保存的修改。放弃这些修改并继续？');
}

window.addEventListener('beforeunload', event => {
  if (dirty) event.preventDefault();
});

function updateLocation() {
  const url = new URL(location.href);
  url.search = '';
  if (task) url.searchParams.set('task', task.id);
  if (frame) url.searchParams.set('image', frame.id);
  history.replaceState(null, '', url);
}

async function showCatalog() {
  const {tasks} = await api('/api/tasks');
  const list = byId('task-list');
  list.replaceChildren();
  for (const item of tasks) {
    const row = make('li');
    const select = button(item.name, () => run(() => openTask(item.id)));
    select.append(make('span', item.done + ' / ' + item.total + ' 已完成'));
    row.append(select);
    list.append(row);
  }
  if (!tasks.length) list.append(make('li', '导入一个 YOLO OBB ZIP 开始审核。'));
  task = frame = image = undefined;
  dirty = false;
  byId('catalog').hidden = false;
  byId('workbench').hidden = true;
  updateLocation();
}

async function openTask(id, preferred) {
  const nextTask = await api(taskPath(id));
  if (!nextTask.images.length) throw new Error('此任务没有图片');
  const chosen = nextTask.images.find(item => item.id === preferred)
    || nextTask.images.find(item => item.status === 'pending' || item.status === 'in_progress')
    || nextTask.images[0];
  task = nextTask;
  byId('catalog').hidden = true;
  byId('workbench').hidden = false;
  byId('task-name').textContent = task.name;
  byId('export-download').href = taskPath() + '/export';
  byId('filter').value = 'all';
  options(byId('draw-class'), task.classes.map((name, index) => [index, name]));
  options(byId('box-class'), task.classes.map((name, index) => [index, name]));
  await showFrame(chosen.id);
  message('');
}

function options(select, values) {
  select.replaceChildren(...values.map(([value, label]) => new Option(label, value)));
}

function selectedBox() {
  return frame?.boxes.find(box => box.id === selected);
}

function visibleSelectedBox() {
  return hiddenBoxes.has(selected) ? undefined : selectedBox();
}

function toggleBoxVisibility(id) {
  if (hiddenBoxes.delete(id)) {
    byId('layer-labels').checked = true;
  } else {
    hiddenBoxes.add(id);
  }
}

function boxName(box, index) {
  return /^g\d+$/i.test(box.id) ? box.id.toUpperCase() : '新增框 ' + (index + 1);
}

function renderFrames() {
  const list = byId('frame-list');
  const filter = byId('filter').value;
  list.replaceChildren();
  const visible = task.images.filter(item => filter === 'all' || item.status === filter);
  for (const item of visible) {
    const row = make('li');
    const select = button(item.name, () => navigate(item.id));
    select.setAttribute('aria-pressed', String(item.id === frame?.id));
    select.append(make('span', statuses[item.status]));
    row.append(select);
    list.append(row);
  }
  if (!visible.length) list.append(make('li', '没有符合筛选的图片'));
  const done = task.images.filter(item => item.status === 'done' || item.status === 'negative').length;
  byId('progress').textContent = done + ' / ' + task.images.length + ' 已完成';
  const index = task.images.findIndex(item => item.id === frame?.id);
  byId('prev').disabled = index <= 0;
  byId('next').disabled = index >= task.images.length - 1;
  if (list.querySelector('[aria-pressed=true]')) list.querySelector('[aria-pressed=true]').scrollIntoView({block: 'nearest', container: 'nearest'});
}

async function showFrame(id) {
  frame = await api(framePath(id));
  originalBoxes = frame.original_boxes || structuredClone(frame.boxes);
  if (!hiddenBoxesByFrame.has(id)) hiddenBoxesByFrame.set(id, new Set());
  hiddenBoxes = hiddenBoxesByFrame.get(id);
  selected = highlighted = gesture = drawing = undefined;
  dirty = false;
  byId('frame-name').textContent = frame.name;
  byId('frame-split').textContent = frame.split;
  byId('status').value = frame.status;
  byId('frame-note').value = frame.note || '';
  byId('preview').hidden = !frame.has_preview;
  byId('preview').href = framePath() + '/preview';
  byId('image-message').hidden = true;
  byId('canvas-wrap').scrollTo(0, 0);
  const pending = new Image();
  image = undefined;
  pending.onload = () => {
    if (frame?.id !== id) return;
    image = pending;
    sizeCanvas();
  };
  pending.onerror = () => {
    if (frame?.id !== id) return;
    byId('image-message').textContent = '这张图片载入失败';
    byId('image-message').hidden = false;
  };
  pending.src = framePath() + '/content';
  sizeCanvas();
  renderFrames();
  renderLabels();
  renderEvents();
  renderHistory();
  updateLocation();
  message('');
}

function navigate(id) {
  if (!frame || id === frame.id || busy || !canLeave()) return;
  run(() => showFrame(id));
}

function neighbor(offset) {
  const index = task.images.findIndex(item => item.id === frame.id);
  return task.images[index + offset];
}

function renderLabels(focusBoxId) {
  const list = byId('box-list');
  list.replaceChildren();
  frame.boxes.forEach((box, index) => {
    const name = boxName(box, index) + ' · ' + task.classes[box.class_id];
    const hidden = hiddenBoxes.has(box.id);
    const row = make('div', null, 'box-row');
    const link = button(name, () => {
      byId('layer-labels').checked = true;
      setSelection(box.id);
      focusPoints(box.points);
    });
    link.className = 'box-select';
    link.disabled = hidden;
    link.setAttribute('aria-pressed', String(box.id === selected));
    const visibility = button(hidden ? '显示' : '隐藏', () => {
      if (!hidden && selected === box.id) selected = undefined;
      toggleBoxVisibility(box.id);
      renderLabels(box.id);
      draw();
    });
    visibility.className = 'box-visibility';
    visibility.setAttribute('aria-label', (hidden ? '显示 ' : '隐藏 ') + name);
    row.append(link, visibility);
    list.append(row);
    if (box.id === focusBoxId) visibility.focus({preventScroll: true});
  });
  byId('box-count').textContent = frame.boxes.length;
  const box = selectedBox();
  byId('box-editor').hidden = !box;
  if (box) {
    const hidden = hiddenBoxes.has(box.id);
    byId('selected-box-visibility').textContent = hidden ? '显示所选框' : '隐藏所选框';
    byId('box-class').disabled = byId('box-angle').disabled = byId('delete-box').disabled = hidden;
    byId('box-class').value = box.class_id;
    byId('box-angle').value = Math.round(geometry(box).angle * 1800 / Math.PI) / 10;
  }
}

function renderEvents() {
  const list = byId('event-list');
  list.replaceChildren();
  updateEventCount();
  if (!frame.events.length) {
    list.append(make('p', '该图片没有已导入的分歧事件。'));
  }
  for (const event of frame.events) {
    const item = make('article', null, 'event');
    const title = (eventKinds[event.kind] || event.kind) + ' · ' + (event.box_id || event.id);
    const select = make('select');
    const values = [['', '待判定'], ...decisions.map(value => [value, value])];
    if (event.decision && !decisions.includes(event.decision)) values.push([event.decision, event.decision]);
    options(select, values);
    select.value = event.decision || '';
    select.addEventListener('change', () => { event.decision = select.value; updateEventCount(); markDirty(); });
    const note = make('textarea');
    note.rows = 2;
    note.value = event.note || '';
    note.addEventListener('input', () => { event.note = note.value; markDirty(); });
    const detail = [event.head, event.class_name, typeof event.score === 'number' ? event.score.toFixed(3) : event.score,
      event.other_box_id, event.other_class_name, event.probiou == null || event.probiou === '' ? '' : 'ProbIoU ' + event.probiou].filter(Boolean);
    item.append(button(title, () => locateEvent(event)), make('small', detail.join(' · ')),
      field('人工判定', select), field('备注', note));
    list.append(item);
  }
}

function updateEventCount() {
  const total = frame.events.length;
  const pending = frame.events.filter(event => !event.decision).length;
  byId('event-count').textContent = total ? '待判定 ' + pending + ' / ' + total : '0';
}

function locateEvent(event) {
  const head = event.head === 'one2many' ? 'one2many' : 'one2one';
  const other = head === 'one2one' ? 'one2many' : 'one2one';
  const ids = {[head]: event.box_id, [other]: event.other_box_id};
  let focus;
  for (const h of [head, other]) {
    if (!ids[h]) continue;
    byId('layer-' + h).checked = true;
    const proposal = frame.predictions?.[h]?.find(item => String(item.id) === String(ids[h]));
    focus ??= proposal?.points;
  }
  highlighted = ids;
  if (focus) {
    focusPoints(focus);
    draw();
  } else if (frame.has_preview) {
    window.open(framePath() + '/preview', '_blank', 'noopener');
  } else {
    message('该事件没有对应的可定位预测框。', true);
  }
}

function renderHistory() {
  const list = byId('history');
  list.replaceChildren();
  const history = frame.history || [];
  byId('history-count').textContent = history.length;
  if (!history.length) list.append(make('p', '暂无保存历史。'));
  for (const entry of history) {
    const item = make('details', null, 'history-entry');
    const heading = [entry.timestamp || entry.time || entry.created_at, entry.action || entry.kind || entry.type || '保存'].filter(Boolean);
    item.append(make('summary', heading.join(' · ')), make('pre', JSON.stringify(entry, null, 2)));
    list.append(item);
  }
}

async function saveFrame(finish = false) {
  if (!frame) return;
  const status = finish && !['negative', 'uncertain'].includes(byId('status').value) ? 'done' : byId('status').value;
  if (status === 'negative' && frame.boxes.length) throw new Error('确认无目标前，请删除当前图片的已有标注。');
  const next = finish ? neighbor(1) : null;
  const payload = {
    version: frame.version, status, note: byId('frame-note').value,
    boxes: frame.boxes, events: frame.events.map(event => ({id: event.id, decision: event.decision || '', note: event.note || ''})),
  };
  frame = await api(framePath(), {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const summary = task.images.find(item => item.id === frame.id);
  if (summary) summary.status = frame.status;
  dirty = false;
  byId('status').value = frame.status;
  renderFrames();
  renderLabels();
  renderHistory();
  draw();
  if (next) await showFrame(next.id);
  message(next ? '上一张已保存' : '当前图片已保存');
}

function sizeCanvas() {
  if (!frame || byId('workbench').hidden) return;
  canvas.width = frame.width;
  canvas.height = frame.height;
  const wrap = byId('canvas-wrap');
  const fit = Math.min(wrap.clientWidth / frame.width, wrap.clientHeight / frame.height);
  const scale = Math.max(.01, fit * Number(byId('zoom').value));
  canvas.style.width = (frame.width * scale) + 'px';
  canvas.style.height = (frame.height * scale) + 'px';
  draw();
}

function normalize(points) { return points.map(([x, y]) => [x / canvas.width, y / canvas.height]); }
function pixels(points) { return points.map(([x, y]) => [x * canvas.width, y * canvas.height]); }
function midpoint(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function subtraction(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
function visualPixels() { return canvas.width / Math.max(canvas.getBoundingClientRect().width, 1); }

function geometry(box) {
  const [a, b, c, d] = pixels(box.points);
  const [dx, dy] = subtraction(b, a);
  return {center: midpoint(a, c), width: distance(a, b), height: distance(a, d), angle: Math.atan2(dy, dx)};
}

function rectangle({center, width, height, angle}) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return [[-width/2, -height/2], [width/2, -height/2], [width/2, height/2], [-width/2, height/2]]
    .map(([x, y]) => [center[0] + x*cos - y*sin, center[1] + x*sin + y*cos]);
}

function fits(points) {
  return points.every(([x, y]) => x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height);
}

function adjustedPoints(points) {
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  if (Math.max(...xs) - Math.min(...xs) > canvas.width || Math.max(...ys) - Math.min(...ys) > canvas.height) return null;
  const dx = Math.max(-Math.min(...xs), Math.min(0, canvas.width - Math.max(...xs)));
  const dy = Math.max(-Math.min(...ys), Math.min(0, canvas.height - Math.max(...ys)));
  return points.map(([x, y]) => [Math.max(0, Math.min(canvas.width, x + dx)),
    Math.max(0, Math.min(canvas.height, y + dy))]);
}

function focusPoints(points) {
  const center = points.reduce(([x, y], [px, py]) => [x + px/points.length, y + py/points.length], [0, 0]);
  const wrap = byId('canvas-wrap');
  wrap.scrollTo(center[0]*canvas.offsetWidth - wrap.clientWidth/2, center[1]*canvas.offsetHeight - wrap.clientHeight/2);
}

function rotationHandle(points) {
  const top = midpoint(points[0], points[1]);
  const center = midpoint(points[0], points[2]);
  const [dx, dy] = subtraction(top, center);
  const length = Math.hypot(dx, dy) || 1;
  const step = 23 * visualPixels();
  return [top[0] + dx/length*step, top[1] + dy/length*step];
}

function strokeBox(points, ink, label, emphasis = false, dashed = false) {
  const scale = visualPixels();
  context.strokeStyle = ink;
  context.fillStyle = ink;
  context.lineWidth = (emphasis ? 2.8 : 1.45) * scale;
  context.setLineDash(dashed ? [4*scale, 3*scale] : []);
  context.beginPath();
  for (const [i, point] of points.entries()) {
    if (i === 0) context.moveTo(...point); else context.lineTo(...point);
  }
  context.closePath();
  context.stroke();
  context.setLineDash([]);
  if (!label) return;
  const [x, y] = points.reduce((best, point) => point[1] < best[1] ? point : best);
  context.font = (10 * scale) + 'px sans-serif';
  context.strokeStyle = '#0b1717';
  context.lineWidth = 3*scale;
  context.strokeText(label, x, Math.max(10*scale, y - 2*scale));
  context.fillText(label, x, Math.max(10*scale, y - 2*scale));
}

function drawHandles(points) {
  const scale = visualPixels();
  const handle = rotationHandle(points);
  context.lineWidth = 1.4*scale;
  context.strokeStyle = color.selected;
  context.fillStyle = color.selected;
  context.beginPath();
  context.moveTo(...midpoint(points[0], points[1]));
  context.lineTo(...handle);
  context.stroke();
  context.beginPath();
  context.arc(...handle, 4.4*scale, 0, 2*Math.PI);
  context.fill();
  for (const [x, y] of points) context.fillRect(x - 2.6*scale, y - 2.6*scale, 5.2*scale, 5.2*scale);
}

function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!frame || !image) return;
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (byId('layer-original').checked) {
    for (const box of originalBoxes) {
      if (!hiddenBoxes.has(box.id)) strokeBox(pixels(box.points), color.original, null, false, true);
    }
  }
  for (const head of ['one2one', 'one2many']) {
    if (!byId('layer-' + head).checked) continue;
    for (const box of frame.predictions?.[head] || []) {
      const emphasis = String(highlighted?.[head]) === String(box.id);
      strokeBox(pixels(box.points), color[head], box.id + ' ' + (task.classes[box.class_id] || ''), emphasis, true);
    }
  }
  if (byId('layer-labels').checked) {
    frame.boxes.forEach((box, index) => {
      if (box.id !== selected && !hiddenBoxes.has(box.id)) {
        strokeBox(pixels(box.points), color.label, boxName(box, index) + ' ' + task.classes[box.class_id]);
      }
    });
    const box = visibleSelectedBox();
    if (box) {
      const points = pixels(box.points);
      strokeBox(points, color.selected, task.classes[box.class_id], true);
      drawHandles(points);
    }
  }
  if (drawing) strokeBox(drawing, color.selected, null, true, true);
}

function setSelection(id) {
  selected = id;
  renderLabels();
  draw();
}

function setMode(next) {
  mode = next;
  canvas.style.cursor = mode === 'draw' ? 'crosshair' : 'default';
  byId('select-tool').setAttribute('aria-pressed', String(mode === 'select'));
  byId('draw-tool').setAttribute('aria-pressed', String(mode === 'draw'));
}

function pointer(event) {
  const bounds = canvas.getBoundingClientRect();
  return [(event.clientX - bounds.left)*canvas.width/bounds.width, (event.clientY - bounds.top)*canvas.height/bounds.height];
}

function inside(point, polygon) {
  let result = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [x, y] = polygon[i], [xx, yy] = polygon[j];
    if ((y > point[1]) !== (yy > point[1]) && point[0] < (xx - x)*(point[1] - y)/(yy - y) + x) result = !result;
  }
  return result;
}

function chooseGesture(point) {
  if (mode === 'draw') return {kind: 'draw', start: point};
  if (!byId('layer-labels').checked) return null;
  const current = visibleSelectedBox();
  if (current) {
    const points = pixels(current.points);
    const reach = 9*visualPixels();
    const currentGeometry = geometry(current);
    if (distance(point, rotationHandle(points)) < reach) return {kind: 'rotate', start: point, box: current, geometry: currentGeometry};
    const corner = points.findIndex(pos => distance(point, pos) < reach);
    if (corner >= 0) return {kind: 'resize', start: point, box: current, geometry: currentGeometry, opposite: points[(corner+2)%4]};
  }
  const hit = frame.boxes.findLast(box => !hiddenBoxes.has(box.id) && inside(point, pixels(box.points)));
  setSelection(hit?.id);
  if (!hit) return null;
  return {kind: 'move', start: point, box: hit, points: rectangle(geometry(hit))};
}

function draftRectangle(start, point) {
  const [x, y] = point.map((value, axis) => Math.max(0, Math.min(axis ? canvas.height : canvas.width, value)));
  const right = Math.max(start[0], x), left = Math.min(start[0], x);
  const bottom = Math.max(start[1], y), top = Math.min(start[1], y);
  return [[left, top], [right, top], [right, bottom], [left, bottom]];
}

function resized(operation, point) {
  const {geometry: initial, opposite} = operation;
  const [dx, dy] = subtraction(point, opposite);
  const c = Math.cos(initial.angle), s = Math.sin(initial.angle);
  const width = Math.abs(dx*c + dy*s), height = Math.abs(-dx*s + dy*c);
  if (width < 3 || height < 3) return null;
  return rectangle({center: midpoint(opposite, point), width, height, angle: initial.angle});
}

function alterGesture(operation, point) {
  const delta = subtraction(point, operation.start);
  if (operation.kind === 'move') return adjustedPoints(operation.points.map(([x,y]) => [x+delta[0],y+delta[1]]));
  if (operation.kind === 'resize') return resized(operation, point);
  const base = operation.geometry;
  const [tx, ty] = subtraction(operation.start, base.center);
  const [x, y] = subtraction(point, base.center);
  return adjustedPoints(rectangle({...base, angle: base.angle + Math.atan2(y,x) - Math.atan2(ty,tx)}));
}

canvas.addEventListener('pointerdown', event => {
  if (busy || event.button !== 0 || !image) return;
  gesture = chooseGesture(pointer(event));
  if (gesture) {
    gesture.original = gesture.box?.points;
    canvas.setPointerCapture(event.pointerId);
  }
});

canvas.addEventListener('pointermove', event => {
  if (!gesture) return;
  const point = pointer(event);
  if (!gesture.changed && distance(point, gesture.start) < 2*visualPixels()) return;
  if (gesture.kind === 'draw') drawing = draftRectangle(gesture.start, point);
  else {
    const points = alterGesture(gesture, point);
    if (!points || !fits(points)) return;
    gesture.box.points = normalize(points);
  }
  gesture.changed = true;
  draw();
});

canvas.addEventListener('pointerup', () => {
  if (!gesture) return;
  if (gesture.kind === 'draw' && drawing) {
    const min = 5 * visualPixels();
    if (distance(drawing[0], drawing[1]) >= min && distance(drawing[0], drawing[3]) >= min) {
      const box = {id: crypto.randomUUID(), class_id: Number(byId('draw-class').value), points: normalize(drawing)};
      frame.boxes.push(box);
      byId('layer-labels').checked = true;
      setSelection(box.id);
      markDirty();
    }
  } else if (gesture.changed) markDirty();
  gesture = drawing = undefined;
  renderLabels();
  draw();
});

canvas.addEventListener('pointercancel', () => {
  if (gesture?.box) gesture.box.points = gesture.original;
  gesture = drawing = undefined;
  draw();
});

byId('selected-box-visibility').addEventListener('click', () => {
  if (!selectedBox() || busy) return;
  toggleBoxVisibility(selected);
  renderLabels();
  draw();
});

byId('box-class').addEventListener('change', event => {
  const box = visibleSelectedBox();
  if (!box || busy) return;
  box.class_id = Number(event.target.value);
  markDirty();
  renderLabels();
  draw();
});

byId('box-angle').addEventListener('change', event => {
  const box = visibleSelectedBox();
  if (!box || busy || !Number.isFinite(event.target.valueAsNumber)) return;
  const points = adjustedPoints(rectangle({...geometry(box), angle: event.target.valueAsNumber * Math.PI/180}));
  if (!points) { message('该角度下的框无法放入图片内。请缩小框再调整。', true); return; }
  box.points = normalize(points);
  markDirty();
  renderLabels();
  draw();
});

function deleteBox() {
  const box = visibleSelectedBox();
  if (!box || busy) return;
  frame.boxes = frame.boxes.filter(item => item.id !== box.id);
  setSelection(undefined);
  markDirty();
}

byId('delete-box').addEventListener('click', deleteBox);
byId('select-tool').addEventListener('click', () => setMode('select'));
byId('draw-tool').addEventListener('click', () => setMode('draw'));
byId('status').addEventListener('change', markDirty);
byId('frame-note').addEventListener('input', markDirty);
byId('save').addEventListener('click', () => run(() => saveFrame()));
byId('done-next').addEventListener('click', () => run(() => saveFrame(true)));
byId('filter').addEventListener('change', renderFrames);
byId('prev').addEventListener('click', () => { const item = neighbor(-1); if (item) navigate(item.id); });
byId('next').addEventListener('click', () => { const item = neighbor(1); if (item) navigate(item.id); });
byId('back').addEventListener('click', () => { if (canLeave()) run(async () => { await showCatalog(); message(''); }); });
byId('export').addEventListener('click', () => {
  if (dirty) {
    message('请先保存当前图片，再导出审核结果。', true);
    return;
  }
  run(async () => {
    const counts = await api(taskPath() + '/export-summary');
    const labels = {total: '图片总数', done: '已完成', remaining: '其余图片',
      added: '新增框操作', deleted: '删除框操作', modified: '修改框操作'};
    byId('export-summary').replaceChildren(...Object.entries(labels).flatMap(([key, label]) =>
      [make('dt', label), make('dd', counts[key])]));
    byId('export-dialog').showModal();
  });
});
byId('export-download').addEventListener('click', () => byId('export-dialog').close());
for (const id of layerIds) byId(id).addEventListener('change', draw);
byId('zoom').addEventListener('input', sizeCanvas);
new ResizeObserver(sizeCanvas).observe(byId('canvas-wrap'));

document.addEventListener('keydown', event => {
  if (!frame || event.target.matches('input,select,textarea,button') || event.altKey || event.metaKey || event.ctrlKey) return;
  if (event.key === 'Delete') deleteBox();
  if (event.key === 'Escape') setMode('select');
});

byId('import').addEventListener('submit', event => {
  event.preventDefault();
  run(async () => {
    const file = byId('import-zip').files[0];
    const reference = byId('import-review').files[0];
    if (!file) return;
    const name = byId('import-name').value.trim() || file.name.replace(/\.zip$/i, '');
    message('正在导入 ' + name + '…');
    const result = await api('/api/import?name=' + encodeURIComponent(name), zipOptions(file));
    let warning;
    if (reference) {
      try { await api(taskPath(result.id) + '/review', zipOptions(reference)); }
      catch (error) { warning = '数据集已导入，复核材料导入失败。' + error.message; }
    }
    byId('import').reset();
    await openTask(result.id);
    message(warning || '已打开审核任务', Boolean(warning));
  });
});

byId('attach-review').addEventListener('change', event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file || !canLeave()) return;
  run(async () => {
    message('正在载入复核材料…');
    await api(taskPath() + '/review', zipOptions(file));
    await openTask(task.id, frame.id);
    message('复核材料已载入');
  });
});

const parameters = new URLSearchParams(location.search);
run(async () => {
  if (parameters.has('task')) {
    try { await openTask(parameters.get('task'), parameters.get('image')); }
    catch (error) { await showCatalog(); throw error; }
  } else await showCatalog();
});
