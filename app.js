/* =====================================================================
 *  المصحح الآلي OMR — يعمل بالكامل في المتصفح باستخدام OpenCV.js
 *  ---------------------------------------------------------------------
 *  مسار المعالجة (Pipeline):
 *   1) التقاط إطار من الكاميرا إلى Canvas
 *   2) تدرج رمادي ← تمويه Gaussian ← حواف Canny
 *   3) البحث عن أكبر شكل رباعي (حدود الورقة) ← تحويل المنظور (تسوية)
 *   4) Threshold ← استخراج الكنتورات ← تصفية الفقاعات ← فرزها إلى صفوف وأعمدة
 *   5) حساب نسبة البكسلات المظللة لكل فقاعة ← مقارنة بنموذج الإجابة ← الدرجة
 * ===================================================================== */

/* ---------------------------------------------------------------------
 * 1) الإعدادات ونماذج الإجابات (Answer Keys)
 *    رقم الخيار يبدأ من 0:  0 = أ ، 1 = ب ، 2 = ج ، 3 = د
 * ------------------------------------------------------------------- */
const CONFIG = {
  numQuestions: 10,          // عدد الأسئلة في الورقة
  numOptions: 4,             // عدد الخيارات لكل سؤال
  optionLabels: ['أ', 'ب', 'ج', 'د'],
  warpWidth: 600,            // عرض الورقة بعد التسوية (بالبكسل) — يوحّد المقاييس
  detectWidth: 800,          // عرض الصورة المصغّرة المستخدمة لاكتشاف الحواف (للسرعة)
  minPaperAreaRatio: 0.15,   // أقل مساحة للورقة نسبةً لمساحة الصورة
  fillThreshold: 0.45,       // نسبة التظليل الدنيا لاعتبار الفقاعة مظللة
  ambiguityRatio: 0.80,      // إن كان ثاني أعلى تظليل ≥ 80% من الأعلى ⇒ تظليل مزدوج
  rtlOptions: true,          // الخيار «أ» على يمين الصف (ورقة عربية)
  livePreviewMs: 350,        // الفاصل الزمني لمعاينة اكتشاف الورقة لحظياً
};

const EXAMS = {
  entrepreneurship: {
    title: 'أساسيات ريادة الأعمال — اختبار قصير',
    key: [1, 0, 2, 3, 1, 0, 2, 1, 3, 0],
  },
  costAccounting: {
    title: 'محاسبة التكاليف — اختبار قصير',
    key: [2, 2, 0, 1, 3, 1, 0, 2, 3, 1],
  },
};

/* ---------------------------------------------------------------------
 * عناصر الواجهة
 * ------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const captureCanvas = $('captureCanvas');
const btnCapture = $('btnCapture');
const btnSwitch = $('btnSwitch');
const fileInput = $('fileInput');
const examSelect = $('examSelect');
const message = $('message');

let cvReady = false;
let stream = null;
let facingMode = 'environment';   // الكاميرا الخلفية افتراضياً
let busy = false;                 // يمنع تداخل المعالجة الحية مع التصحيح
let previewTimer = null;

/* =====================================================================
 *  تحميل OpenCV.js
 *  بعض إصدارات OpenCV.js تُعرّف cv كـ Promise وبعضها تستدعي
 *  onRuntimeInitialized — نتعامل مع الحالتين.
 * ===================================================================== */
// نجرّب عدة مصادر بالترتيب؛ إن تعذّر أحدها (حجب، بطء، انقطاع) ننتقل للتالي
const OPENCV_SOURCES = [
  'opencv.js', // النسخة المرفوعة مع التطبيق (الأسرع والأضمن على GitHub Pages)
  'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js',
  'https://unpkg.com/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js',
  'https://docs.opencv.org/4.10.0/opencv.js',
];

function loadScript(src, timeoutMs) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.async = true;
    const timer = setTimeout(() => { s.remove(); reject(new Error('timeout')); }, timeoutMs);
    s.onload = () => { clearTimeout(timer); resolve(); };
    s.onerror = () => { clearTimeout(timer); s.remove(); reject(new Error('network')); };
    document.head.appendChild(s);
  });
}

// انتظار اكتمال تهيئة WebAssembly
// تنبيه: لا نستخدم await cv مباشرة؛ فكائن Emscripten يملك دالة then
// تُعيد الكائن نفسه، مما يسبب حلقة لا نهائية تجمّد الصفحة.
function waitForRuntime(timeoutMs) {
  return new Promise((resolve, reject) => {
    if (typeof window.cv === 'undefined') return reject(new Error('cv undefined'));
    const timer = setTimeout(() => reject(new Error('init timeout')), timeoutMs);
    const done = (m) => { clearTimeout(timer); window.cv = m; resolve(); }; // resolve بلا قيمة
    if (cv.Mat) done(cv);
    else if (typeof cv.then === 'function') cv.then(done);
    else cv.onRuntimeInitialized = () => done(cv);
  });
}

async function loadOpenCv() {
  for (let i = 0; i < OPENCV_SOURCES.length; i++) {
    const src = OPENCV_SOURCES[i];
    setBadge(`جارٍ تحميل OpenCV… (${i + 1}/${OPENCV_SOURCES.length})`, 'wait');
    try {
      await loadScript(src, 60000);
      await waitForRuntime(60000);
      cvReady = true;
      setBadge('OpenCV جاهز', 'ok');
      showMessage('');
      btnCapture.disabled = !stream;
      startLivePreview();
      return;
    } catch (e) {
      console.warn('فشل تحميل OpenCV من', src, e.message);
      try { delete window.cv; } catch (_) { window.cv = undefined; }
    }
  }
  setBadge('تعذّر تحميل OpenCV', 'err');
  showMessage('تعذّر تحميل مكتبة OpenCV.js من كل المصادر — تحقق من الإنترنت ثم أعد تحميل الصفحة.', 'err');
}

/* =====================================================================
 *  الكاميرا
 * ===================================================================== */
async function startCamera() {
  stopCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    showMessage('المتصفح لا يدعم الكاميرا — استخدم زر «رفع صورة».', 'err');
    return;
  }
  try {
    // ideal بدلاً من exact: إن لم توجد كاميرا خلفية (لابتوب) تُستخدم الأمامية
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    btnCapture.disabled = !cvReady;
    showMessage('');
  } catch (err) {
    showMessage('لم يُسمح بالوصول للكاميرا (يتطلب HTTPS). يمكنك رفع صورة بدلاً من ذلك.', 'err');
  }
}
function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

/* =====================================================================
 *  أدوات مساعدة لإدارة الذاكرة
 *  كائنات cv.Mat تُحجز في ذاكرة WebAssembly ولا يحررها جامع القمامة،
 *  لذا نسجّلها في «متعقّب» ونحذفها كلها في النهاية.
 * ===================================================================== */
function createTracker() {
  const list = [];
  return {
    add: (m) => { list.push(m); return m; },
    free: () => { list.forEach((m) => { try { m.delete(); } catch (_) {} }); list.length = 0; },
  };
}

/* =====================================================================
 *  المرحلة 2: المعالجة الأولية
 *  ---------------------------------------------------------------------
 *  - Grayscale: Y = 0.299R + 0.587G + 0.114B  (نحتاج قناة واحدة للإضاءة)
 *  - GaussianBlur: التفاف (Convolution) مع نواة غاوسية 5×5
 *        G(x,y) = (1 / 2πσ²) · e^(−(x²+y²)/2σ²)
 *    يقلل الضوضاء والتفاصيل الدقيقة كي لا تُكتشف كحواف زائفة.
 *  - Canny: يحسب تدرج الشدة (Sobel) ثم يُبقي القمم المحلية فقط
 *    (Non-max suppression) ثم يطبّق عتبتين (Hysteresis):
 *    الحواف > 200 قوية، وبين 75 و200 تُقبل فقط إن اتصلت بحافة قوية.
 * ===================================================================== */
function preprocess(src, t) {
  const gray = t.add(new cv.Mat());
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  const blurred = t.add(new cv.Mat());
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const edges = t.add(new cv.Mat());
  cv.Canny(blurred, edges, 75, 200);

  // تمدد بسيط (Dilation) لسدّ الفجوات الصغيرة في خط حدود الورقة
  const kernel = t.add(cv.Mat.ones(3, 3, cv.CV_8U));
  cv.dilate(edges, edges, kernel);

  return { gray, edges };
}

/* =====================================================================
 *  المرحلة 3: إيجاد أكبر مستطيل (حدود الورقة)
 *  ---------------------------------------------------------------------
 *  - نستخرج الكنتورات الخارجية من صورة الحواف ونرتبها تنازلياً حسب المساحة.
 *  - لكل كنتور نطبق approxPolyDP (خوارزمية Douglas–Peucker):
 *    تبسّط المنحنى إلى مضلع بحيث لا يبعد أي نقطة أصلية عن المضلع
 *    أكثر من ε = 2% من محيط الكنتور.
 *  - أول مضلع محدّب له 4 رؤوس ومساحته كافية = الورقة.
 * ===================================================================== */
function findPaperQuad(edges, t, maxAreaRatio = 1.01, mode = cv.RETR_EXTERNAL) {
  const contours = t.add(new cv.MatVector());
  const hierarchy = t.add(new cv.Mat());
  cv.findContours(edges, contours, hierarchy, mode, cv.CHAIN_APPROX_SIMPLE);

  const minArea = edges.rows * edges.cols * CONFIG.minPaperAreaRatio;
  const candidates = [];
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const area = cv.contourArea(c);
    const maxArea = edges.rows * edges.cols * maxAreaRatio;
    if (area >= minArea && area <= maxArea) candidates.push({ c, area });
    else c.delete();
  }
  candidates.sort((a, b) => b.area - a.area);

  let quad = null;
  for (const { c } of candidates) {
    if (!quad) {
      const peri = cv.arcLength(c, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(c, approx, 0.02 * peri, true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const d = approx.data32S;
        quad = [
          { x: d[0], y: d[1] }, { x: d[2], y: d[3] },
          { x: d[4], y: d[5] }, { x: d[6], y: d[7] },
        ];
      }
      approx.delete();
    }
    c.delete();
  }
  return quad;
}

/* ---------------------------------------------------------------------
 *  ترتيب الزوايا الأربع: أعلى‑يسار، أعلى‑يمين، أسفل‑يمين، أسفل‑يسار
 *  الفكرة الرياضية:
 *   - مجموع (x + y): أصغره عند أعلى‑اليسار وأكبره عند أسفل‑اليمين.
 *   - الفرق  (y − x): أصغره عند أعلى‑اليمين وأكبره عند أسفل‑اليسار.
 * ------------------------------------------------------------------- */
function orderCorners(pts) {
  const bySum = [...pts].sort((a, b) => (a.x + a.y) - (b.x + b.y));
  const byDiff = [...pts].sort((a, b) => (a.y - a.x) - (b.y - b.x));
  return { tl: bySum[0], br: bySum[3], tr: byDiff[0], bl: byDiff[3] };
}

const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);

/* ---------------------------------------------------------------------
 *  تحويل المنظور (Perspective / Homography)
 *  نبحث عن مصفوفة H (3×3) تحقق لكل زاوية:
 *        [x' y' w]ᵀ = H · [x y 1]ᵀ   ثم   (x'/w , y'/w)
 *  getPerspectiveTransform تحلّ هذا النظام من 4 أزواج نقاط (8 معادلات،
 *  8 مجاهيل)، ثم warpPerspective تُعيد أخذ عينات كل بكسل وفق H.
 *  أبعاد الناتج تُحسب من أطوال أضلاع الرباعي للحفاظ على نسبة الأبعاد
 *  (كي تبقى الدوائر دوائر)، ثم نوحّد العرض إلى warpWidth.
 * ------------------------------------------------------------------- */
function warpPaper(src, corners, t) {
  const { tl, tr, br, bl } = corners;
  const wPx = Math.max(dist(tl, tr), dist(bl, br));
  const hPx = Math.max(dist(tl, bl), dist(tr, br));
  const W = CONFIG.warpWidth;
  const H = Math.round(hPx * (W / wPx));

  const srcTri = t.add(cv.matFromArray(4, 1, cv.CV_32FC2,
    [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]));
  const dstTri = t.add(cv.matFromArray(4, 1, cv.CV_32FC2,
    [0, 0, W - 1, 0, W - 1, H - 1, 0, H - 1]));

  const M = t.add(cv.getPerspectiveTransform(srcTri, dstTri));
  const warped = t.add(new cv.Mat());
  cv.warpPerspective(src, warped, M, new cv.Size(W, H),
    cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
  return warped;
}

/* ---------------------------------------------------------------------
 *  اكتشاف الورقة على صورة مصغّرة (أسرع) ثم إعادة الإحداثيات للحجم الأصلي
 * ------------------------------------------------------------------- */
function detectPaper(src, t, edgesDebugCanvas) {
  const scale = Math.min(1, CONFIG.detectWidth / src.cols);
  const small = t.add(new cv.Mat());
  cv.resize(src, small, new cv.Size(Math.round(src.cols * scale), Math.round(src.rows * scale)), 0, 0, cv.INTER_AREA);

  const { edges } = preprocess(small, t);
  if (edgesDebugCanvas) cv.imshow(edgesDebugCanvas, edges);

  const quad = findPaperQuad(edges, t);
  if (!quad) return null;
  return quad.map((p) => ({ x: p.x / scale, y: p.y / scale }));
}

/* ---------------------------------------------------------------------
 *  التسوية على مرحلتين:
 *  قد يكون «أكبر مستطيل» في الصورة هو حافة الورقة البيضاء نفسها وليس
 *  الإطار الأسود المطبوع. في هذه الحالة يحيط الإطار بكل الفقاعات، ومع
 *  RETR_EXTERNAL لن نرى إلا الإطار! لذلك نبحث مرة ثانية داخل الورقة
 *  المسوّاة عن مستطيل أصغر (الإطار) ونسوّي عليه.
 * ------------------------------------------------------------------- */
function refineToFrame(warped, t) {
  const { edges } = preprocess(warped, t);
  // نتجاهل أي مستطيل يغطي أكثر من 90% من الصورة (أي حافة الصورة نفسها)،
  // ونستخدم RETR_LIST لأن الإطار يقع «داخل» كنتور حافة الصورة فلا يظهر مع EXTERNAL
  const quad = findPaperQuad(edges, t, 0.9, cv.RETR_LIST);
  if (!quad) return warped;   // لا يوجد إطار داخلي ⇒ نحن بالفعل على الإطار
  return warpPaper(warped, orderCorners(quad), t);
}

/* =====================================================================
 *  المرحلة 4: اكتشاف الفقاعات وفرزها
 * ===================================================================== */
function findBubbles(warped, t) {
  const gray = t.add(new cv.Mat());
  cv.cvtColor(warped, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, gray, new cv.Size(3, 3), 0);

  /* Otsu Thresholding: يختار تلقائياً العتبة T التي تعظّم التباين
   * بين الفئتين (الخلفية/الحبر):  σ_b²(T) = ω₀·ω₁·(μ₀ − μ₁)²
   * ونستخدم INV كي يصبح الحبر أبيض (255) والورق أسود (0)،
   * فتصبح نسبة البكسلات البيضاء داخل الفقاعة = نسبة التظليل. */
  const thresh = t.add(new cv.Mat());
  cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);

  // نمحو هامشاً رفيعاً عند الأطراف كي لا يلتصق إطار الورقة بأي شيء
  const m = Math.round(CONFIG.warpWidth * 0.03);
  cv.rectangle(thresh, new cv.Point(0, 0), new cv.Point(thresh.cols - 1, thresh.rows - 1),
    new cv.Scalar(0), m * 2);

  const contours = t.add(new cv.MatVector());
  const hierarchy = t.add(new cv.Mat());
  // RETR_EXTERNAL: الكنتور الخارجي لكل فقاعة فقط (يتجاهل الثقوب/الحروف داخلها)
  cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  /* تصفية هندسية للفقاعات:
   *  - الحجم: بين 2.5% و 9% من عرض الورقة (يستبعد النصوص الصغيرة والإطارات الكبيرة)
   *  - نسبة العرض/الارتفاع ≈ 1 (شكل دائري/مربع)
   *  - الاستدارة (Circularity) = 4π·المساحة / المحيط²  (=1 للدائرة المثالية) */
  const W = thresh.cols;
  let bubbles = [];
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const r = cv.boundingRect(c);
    const ar = r.width / r.height;
    const area = cv.contourArea(c);
    const peri = cv.arcLength(c, true);
    const circularity = peri > 0 ? (4 * Math.PI * area) / (peri * peri) : 0;
    c.delete();

    if (r.width >= W * 0.025 && r.width <= W * 0.09 &&
        ar >= 0.75 && ar <= 1.33 && circularity > 0.6) {
      bubbles.push({ idx: i, x: r.x, y: r.y, w: r.width, h: r.height,
        cx: r.x + r.width / 2, cy: r.y + r.height / 2 });
    }
  }

  // إبقاء الفقاعات ذات الحجم المتقارب فقط (±30% من الوسيط) لإزالة الشوائب
  if (bubbles.length) {
    const ws = bubbles.map((b) => b.w).sort((a, b) => a - b);
    const med = ws[Math.floor(ws.length / 2)];
    bubbles = bubbles.filter((b) => b.w > med * 0.7 && b.w < med * 1.3);
  }

  return { thresh, contours, bubbles };
}

/* ---------------------------------------------------------------------
 *  خوارزمية الفرز إلى شبكة (أسئلة × خيارات)
 *  ---------------------------------------------------------------------
 *  الخطوة 1 — فرز رأسي: نرتب كل الفقاعات تصاعدياً حسب مركزها الرأسي cy.
 *  الخطوة 2 — تجميع إلى صفوف: نمرّ على القائمة المرتبة؛ إذا كان الفرق بين
 *     cy للفقاعة الحالية ومتوسط cy للصف الحالي أصغر من نصف قطر الفقاعة
 *     تقريباً (0.6 × الارتفاع الوسيط) فهي في نفس الصف، وإلا نبدأ صفاً جديداً.
 *     (هذا أمتن من التقسيم الثابت كل 4 عناصر لأنه يتحمّل الميلان الطفيف.)
 *  الخطوة 3 — نُبقي الصفوف التي تحتوي بالضبط numOptions فقاعة.
 *  الخطوة 4 — فرز أفقي داخل كل صف: حسب cx تنازلياً للورقة العربية
 *     (أ على اليمين) أو تصاعدياً للورقة اللاتينية.
 *  النتيجة: grid[q][o] = الفقاعة الخاصة بالسؤال q والخيار o.
 * ------------------------------------------------------------------- */
function sortBubblesIntoGrid(bubbles) {
  const sorted = [...bubbles].sort((a, b) => a.cy - b.cy);
  const hs = sorted.map((b) => b.h).sort((a, b) => a - b);
  const tol = (hs[Math.floor(hs.length / 2)] || 20) * 0.6;

  const rows = [];
  for (const b of sorted) {
    const row = rows[rows.length - 1];
    if (row) {
      const meanY = row.reduce((s, x) => s + x.cy, 0) / row.length;
      if (Math.abs(b.cy - meanY) < tol) { row.push(b); continue; }
    }
    rows.push([b]);
  }

  const valid = rows.filter((r) => r.length === CONFIG.numOptions);
  valid.forEach((r) => r.sort((a, b) => CONFIG.rtlOptions ? b.cx - a.cx : a.cx - b.cx));
  return { grid: valid, rawRows: rows };
}

/* =====================================================================
 *  المرحلة 5: قياس التظليل واختيار الإجابة
 *  ---------------------------------------------------------------------
 *  لكل فقاعة: نرسم قناعاً (Mask) ممتلئاً بشكلها، ثم نحسب:
 *      ratio = |Mask ∩ Thresh| / |Mask|
 *  أي عدد البكسلات المظللة داخل الفقاعة مقسوماً على مساحتها.
 *  الفقاعة الفارغة (إطار فقط) ≈ 0.15–0.35 ، والمظللة ≈ 0.7–1.0
 *  القرار:
 *   - أعلى نسبة < fillThreshold        ⇒ لم يُجب
 *   - ثاني أعلى ≥ 80% من الأعلى وفوق العتبة ⇒ تظليل مزدوج (يُعدّ خطأ)
 *   - غير ذلك                           ⇒ الخيار ذو النسبة الأعلى
 * ===================================================================== */
function measureFill(thresh, contours, bubble, t) {
  // نعمل داخل منطقة الفقاعة فقط (ROI) لتسريع الحساب
  const rect = new cv.Rect(bubble.x, bubble.y, bubble.w, bubble.h);
  const mask = cv.Mat.zeros(thresh.rows, thresh.cols, cv.CV_8UC1);
  cv.drawContours(mask, contours, bubble.idx, new cv.Scalar(255), -1);

  const maskRoi = mask.roi(rect);
  const threshRoi = thresh.roi(rect);
  const both = new cv.Mat();
  cv.bitwise_and(threshRoi, threshRoi, both, maskRoi);

  const total = cv.countNonZero(maskRoi);
  const filled = cv.countNonZero(both);

  [mask, maskRoi, threshRoi, both].forEach((m) => m.delete());
  return total ? filled / total : 0;
}

function gradeGrid(grid, thresh, contours, key, t) {
  return grid.map((row, q) => {
    const ratios = row.map((b) => measureFill(thresh, contours, b, t));
    const order = ratios.map((r, i) => [r, i]).sort((a, b) => b[0] - a[0]);
    const [best, second] = order;

    let chosen = -1, status = 'blank';
    if (best[0] >= CONFIG.fillThreshold) {
      if (second && second[0] >= CONFIG.fillThreshold && second[0] >= best[0] * CONFIG.ambiguityRatio) {
        status = 'multi';
      } else {
        chosen = best[1];
        status = chosen === key[q] ? 'correct' : 'wrong';
      }
    }
    return { q, chosen, correct: key[q], status, ratios, row };
  });
}

/* ---------------------------------------------------------------------
 *  رسم النتيجة فوق الورقة المسوّاة
 *  أخضر = إجابة صحيحة ، أحمر = إجابة خاطئة ، أزرق = موضع الإجابة الصحيحة
 * ------------------------------------------------------------------- */
function drawResults(warped, results) {
  const GREEN = new cv.Scalar(26, 157, 85, 255);
  const RED = new cv.Scalar(214, 58, 58, 255);
  const BLUE = new cv.Scalar(31, 111, 235, 255);
  const ORANGE = new cv.Scalar(201, 138, 0, 255);

  for (const r of results) {
    const radius = (b) => Math.round(Math.max(b.w, b.h) / 2 + 3);
    const center = (b) => new cv.Point(Math.round(b.cx), Math.round(b.cy));
    const correctB = r.row[r.correct];

    if (r.status === 'correct') {
      cv.circle(warped, center(correctB), radius(correctB), GREEN, 3);
    } else {
      if (r.chosen >= 0) cv.circle(warped, center(r.row[r.chosen]), radius(r.row[r.chosen]), RED, 3);
      if (r.status === 'multi' || r.status === 'blank') {
        r.row.forEach((b) => cv.circle(warped, center(b), radius(b), ORANGE, 1));
      }
      cv.circle(warped, center(correctB), radius(correctB), BLUE, 2);
    }
  }
}

/* =====================================================================
 *  الدالة الرئيسية: التقاط ← معالجة ← تصحيح ← عرض
 * ===================================================================== */
function gradeFromCanvas(canvas) {
  const t = createTracker();
  try {
    const exam = EXAMS[examSelect.value];
    if (exam.key.length !== CONFIG.numQuestions) throw new Error('طول نموذج الإجابة لا يطابق عدد الأسئلة.');

    const src = t.add(cv.imread(canvas));

    // (2) + (3) اكتشاف الورقة وتسويتها
    const quadPts = detectPaper(src, t, $('dbgEdges'));
    if (!quadPts) throw new Error('لم يتم العثور على حدود الورقة. تأكد أن الإطار الأسود للورقة ظاهر بالكامل وأن الخلفية مختلفة اللون.');
    let warped = warpPaper(src, orderCorners(quadPts), t);
    warped = refineToFrame(warped, t);

    // (4) الفقاعات
    const { thresh, contours, bubbles } = findBubbles(warped, t);
    cv.imshow($('dbgThresh'), thresh);

    const { grid } = sortBubblesIntoGrid(bubbles);
    if (grid.length !== CONFIG.numQuestions) {
      throw new Error(`تم اكتشاف ${grid.length} صف/سؤال صالح (${bubbles.length} فقاعة) بدلاً من ${CONFIG.numQuestions}. قرّب الكاميرا وحسّن الإضاءة وتجنب الظلال.`);
    }

    // (5) التصحيح
    const results = gradeGrid(grid, thresh, contours, exam.key, t);
    drawResults(warped, results);
    cv.imshow($('resultCanvas'), warped);
    renderResults(results, exam);
    showMessage('تم التصحيح بنجاح ✓', 'ok');
  } catch (e) {
    console.error(e);
    showMessage(e.message || String(e), 'err');
  } finally {
    t.free();   // تحرير ذاكرة WebAssembly
  }
}

/* ---------------------------------------------------------------------
 *  التقاط الإطار الحالي من الفيديو بكامل الدقة
 * ------------------------------------------------------------------- */
function captureFrame() {
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext('2d').drawImage(video, 0, 0);
  return captureCanvas;
}

/* =====================================================================
 *  معاينة حية: رسم حدود الورقة المكتشفة فوق الفيديو كل 350ms
 * ===================================================================== */
const previewCanvas = document.createElement('canvas');
function startLivePreview() {
  clearInterval(previewTimer);
  previewTimer = setInterval(() => {
    if (!cvReady || busy || !stream || video.readyState < 2) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    const s = Math.min(1, 480 / vw);
    previewCanvas.width = Math.round(vw * s);
    previewCanvas.height = Math.round(vh * s);
    previewCanvas.getContext('2d').drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);

    const t = createTracker();
    let quad = null;
    try {
      const src = t.add(cv.imread(previewCanvas));
      quad = detectPaper(src, t, null);
    } catch (_) { /* نتجاهل أخطاء المعاينة */ } finally { t.free(); }
    drawOverlay(quad, previewCanvas.width, previewCanvas.height);
  }, CONFIG.livePreviewMs);
}

function drawOverlay(quad, w, h) {
  const rect = overlay.getBoundingClientRect();
  overlay.width = rect.width; overlay.height = rect.height;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $('camHint').textContent = quad ? 'تم اكتشاف الورقة — اضغط «التقاط وتصحيح»' : 'ضع الورقة كاملة داخل الإطار';
  if (!quad) return;

  // تحويل الإحداثيات من حجم المعاينة إلى حجم العرض مع مراعاة object-fit: cover
  const k = Math.max(rect.width / w, rect.height / h);
  const ox = (rect.width - w * k) / 2, oy = (rect.height - h * k) / 2;
  const { tl, tr, br, bl } = orderCorners(quad);
  ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 4; ctx.fillStyle = 'rgba(34,197,94,.15)';
  ctx.beginPath();
  [tl, tr, br, bl].forEach((p, i) => ctx[i ? 'lineTo' : 'moveTo'](ox + p.x * k, oy + p.y * k));
  ctx.closePath(); ctx.fill(); ctx.stroke();
}

/* =====================================================================
 *  عرض النتائج في الواجهة
 * ===================================================================== */
function renderResults(results, exam) {
  const L = CONFIG.optionLabels;
  const score = results.filter((r) => r.status === 'correct').length;
  const pct = Math.round((score / results.length) * 100);

  $('resultSection').classList.remove('hidden');
  $('scoreValue').textContent = score;
  $('scoreTotal').textContent = 'من ' + results.length;
  $('scoreTitle').textContent = exam.title;
  $('scorePercent').textContent = `النسبة: ${pct}%`;
  document.querySelector('.score-circle').style.setProperty('--pct', pct + '%');

  const statusText = {
    correct: ['✔ صحيحة', 'st-ok'],
    wrong: ['✘ خاطئة', 'st-bad'],
    blank: ['— بدون إجابة', 'st-warn'],
    multi: ['⚠ تظليل مزدوج', 'st-warn'],
  };
  $('answersBody').innerHTML = results.map((r) => {
    const [txt, cls] = statusText[r.status];
    return `<tr>
      <td>${r.q + 1}</td>
      <td>${r.chosen >= 0 ? L[r.chosen] : '—'}</td>
      <td>${L[r.correct]}</td>
      <td class="${cls}">${txt}</td>
    </tr>`;
  }).join('');
  $('resultSection').scrollIntoView({ behavior: 'smooth' });
}

function showMessage(txt, type = '') { message.textContent = txt; message.className = 'message ' + type; }
function setBadge(txt, type) { const b = $('cvStatus'); b.textContent = txt; b.className = 'badge badge-' + type; }

/* =====================================================================
 *  ربط الأحداث
 * ===================================================================== */
function runGrading(canvas) {
  if (!cvReady) return showMessage('انتظر حتى يكتمل تحميل OpenCV.', 'err');
  busy = true;
  showMessage('جارٍ التصحيح…');
  // setTimeout يتيح للواجهة تحديث الرسالة قبل بدء المعالجة الثقيلة
  setTimeout(() => { gradeFromCanvas(canvas); busy = false; }, 30);
}

btnCapture.addEventListener('click', () => {
  if (!stream || video.readyState < 2) return showMessage('الكاميرا غير جاهزة.', 'err');
  runGrading(captureFrame());
});

btnSwitch.addEventListener('click', () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    captureCanvas.width = img.naturalWidth;
    captureCanvas.height = img.naturalHeight;
    captureCanvas.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(img.src);
    runGrading(captureCanvas);
  };
  img.src = URL.createObjectURL(file);
  fileInput.value = '';
});

// تعبئة قائمة النماذج
examSelect.innerHTML = Object.entries(EXAMS)
  .map(([id, ex]) => `<option value="${id}">${ex.title}</option>`).join('');

startCamera();
loadOpenCv();
