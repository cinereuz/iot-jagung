// Corn Kernel Mold Detection Frontend Application

document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const sampleButtonsContainer = document.getElementById('sample-buttons');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results-section');

  let currentResults = null;

  // Load available samples from server
  async function loadSamples() {
    if (!sampleButtonsContainer) return;
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      renderSampleButtons(data.samples);
    } catch (err) {
      console.error('Gagal memuat sampel:', err);
    }
  }

  function renderSampleButtons(samples) {
    if (!sampleButtonsContainer) return;
    sampleButtonsContainer.innerHTML = '';

    if (!samples || samples.length === 0) {
      sampleButtonsContainer.innerHTML = "<span style='color: var(--text-muted); font-size: 0.85rem;'>No local samples found.</span>";
      return;
    }

    samples.forEach((s) => {
      const btn = document.createElement('button');
      btn.className = `btn-sample ${s.category === 'sehat' ? 'healthy' : 'moldy'}`;
      btn.innerHTML = `<span class="dot ${s.category === 'sehat' ? 'green' : 'red'}"></span> ${s.name}`;
      btn.addEventListener('click', () => detectFromSample(s.path));
      sampleButtonsContainer.appendChild(btn);
    });
  }

  // Drag & drop handlers
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      detectFromUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      detectFromUpload(e.target.files[0]);
    }
  });

  // Detect from uploaded file
  async function detectFromUpload(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/detect/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'An error occurred while processing the image.');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  // Detect from server sample path
  async function detectFromSample(samplePath) {
    showLoading(true);
    try {
      const res = await fetch('/api/detect/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_path: samplePath }),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to process the sample.');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  function showLoading(show) {
    if (show) {
      loadingOverlay.classList.add('active');
    } else {
      loadingOverlay.classList.remove('active');
    }
  }

  // BARU -- Helper kecil terpusat untuk menentukan "state" tiap kernel (biji).
  // Dipakai di renderKernelCards() dan renderFeatureTable() supaya logic
  // penentuan warna/label/teks tidak ditulis dobel di dua tempat.
  // Return: { stateClass, label, colorHex }
  function getKernelState(k) {
    if (k.prediction === 'bukan_jagung') {
      // Abu-abu -- menandakan objek ini DIABAIKAN dari penilaian mutu,
      // bukan salah satu dari sehat/terkontaminasi. Warnanya didefinisikan
      // di style.css lewat variabel --notcorn-gray.
      return { stateClass: 'notcorn', label: 'Not Corn', colorHex: 'var(--notcorn-gray)' };
    }
    if (k.is_moldy) {
      return { stateClass: 'moldy', label: 'Mold Contaminated', colorHex: 'var(--mold-red)' };
    }
    return { stateClass: 'healthy', label: 'Healthy', colorHex: 'var(--healthy-green)' };
  }

  // Render results
  function displayResults(data) {
    currentResults = data;
    const summary = data.summary;
    const images = data.images;
    const kernels = data.kernels;

    // BARU -- summary.bukan_jagung mungkin tidak ada kalau responsenya dari
    // pipeline.py versi lama (backward compatible). Default ke 0 kalau kosong.
    const bukanJagungCount = summary.bukan_jagung || 0;
    // BARU -- total biji yang BENAR-BENAR jagung (dipakai untuk kalimat ringkasan
    // yang lebih akurat, karena summary.total_biji sekarang bisa termasuk
    // objek yang bukan jagung).
    const totalJagung = summary.sehat + summary.terkontaminasi;

    // 1. Status Banner
    const statusBanner = document.getElementById('status-banner');
    // style.css sekarang sudah punya rule .status-banner.secondary untuk kasus
    // "semua objek bukan jagung", jadi cukup ganti class-nya saja seperti
    // badge success/warning/danger yang sudah ada.
    statusBanner.className = `status-banner ${summary.badge}`;
    document.getElementById('status-title').textContent = summary.status_mutu;

    // BARU -- kalimat ringkasan sekarang menyebut jumlah bukan_jagung juga,
    // dan pakai totalJagung (bukan summary.total_biji) supaya persentase
    // kontaminasi yang disebut tetap konsisten dengan angka yang dihitung
    // pipeline (kontaminasi dihitung dari objek yang memang jagung saja).
    let descText = '';
    if (totalJagung === 0) {
      descText = 'No corn kernels were detected in this image.';
      if (bukanJagungCount > 0) {
        descText += ` Objects detected as NOT corn: ${bukanJagungCount}.`;
      }
    } else if (bukanJagungCount > totalJagung) {
      descText = `Not Corn: detected ${bukanJagungCount} non-corn object(s) vs only ${totalJagung} corn-like object(s) (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Because non-corn objects predominate, the overall result is classified as Not Corn.`;
    } else {
      descText = `Corn kernels detected: ${totalJagung} (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Contamination rate: ${summary.persen_kontaminasi}%.`;
      if (bukanJagungCount > 0) {
        descText += ` Other objects not recognized as corn kernels: ${bukanJagungCount} (excluded from the quality assessment).`;
      }
    }
    document.getElementById('status-desc').textContent = descText;

    // 2. Metrics Cards
    document.getElementById('metric-total').textContent = summary.total_biji;
    document.getElementById('metric-healthy').textContent = summary.sehat;
    document.getElementById('metric-moldy').textContent = summary.terkontaminasi;
    document.getElementById('metric-rate').textContent = `${summary.persen_kontaminasi}%`;

    // BARU -- kartu "Bukan Jagung" cuma ditampilkan kalau memang ada objek
    // yang terdeteksi bukan jagung, supaya tidak mengganggu tampilan normal.
    const notcornCard = document.getElementById('metric-card-notcorn');
    if (bukanJagungCount > 0) {
      notcornCard.style.display = '';
      document.getElementById('metric-notcorn').textContent = bukanJagungCount;
    } else {
      notcornCard.style.display = 'none';
    }

    // 3. Stage Visualizer Setup
    setupStages(images);

    // 4. Kernel Cards Grid
    renderKernelCards(kernels);

    // 5. Feature Table
    renderFeatureTable(kernels);

    // 6. Hybrid Stepper & Tab Pipeline Inspector Setup
    setupPipelineInspector(data);

    // 7. BARU -- Info kontribusi dataset crowdsourcing (fungsi ada di admin.js)
    if (typeof window.renderStagingNotice === 'function') {
      window.renderStagingNotice(data.staging);
    }

    // Reveal section and smooth scroll
    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  // ================================================================
  // HYBRID STEPPER & TAB PIPELINE INSPECTOR
  // ================================================================
  let activePipelineStep = 1;
  let selectedKernelIndex = 0;
  let isShowAllMode = false;

  const STEP_TITLES = {
    1: 'Input & ROI Standardization',
    2: 'Preprocessing (Grayscale & Blur)',
    3: 'Segmentation (Otsu & K-Means)',
    4: 'Feature Extraction (GLCM & HSV)',
    5: 'SVM Classification & Novelty Gate',
  };

  function opencvHsvToCssRgb(hOpenCv, sOpenCv, vOpenCv) {
    const h = Math.max(0, Math.min(360, Number(hOpenCv) * 2));
    const s = Math.max(0, Math.min(1, Number(sOpenCv) / 255));
    const v = Math.max(0, Math.min(1, Number(vOpenCv) / 255));
    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;
    let r1 = 0, g1 = 0, b1 = 0;
    if (h < 60) { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }
    const r = Math.round((r1 + m) * 255);
    const g = Math.round((g1 + m) * 255);
    const b = Math.round((b1 + m) * 255);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function switchPipelineStep(stepNum) {
    activePipelineStep = Math.max(1, Math.min(5, Number(stepNum)));
    const tabs = document.querySelectorAll('.stepper-tab');
    const panes = document.querySelectorAll('.pipeline-step-pane');
    const selectorBar = document.getElementById('kernel-selector-bar');
    const prevBtn = document.getElementById('btn-step-prev');
    const nextBtn = document.getElementById('btn-step-next');
    const progLabel = document.getElementById('stepper-progress-label');

    tabs.forEach((t) => {
      const s = Number(t.dataset.step);
      t.classList.toggle('active', s === activePipelineStep);
      t.classList.toggle('completed', s < activePipelineStep);
    });

    panes.forEach((p) => {
      const s = Number(p.dataset.stepPane);
      p.classList.toggle('active', s === activePipelineStep);
    });

    if (selectorBar) {
      const showKernelBar = isShowAllMode || activePipelineStep >= 3;
      selectorBar.style.display = showKernelBar ? 'flex' : 'none';
    }

    if (progLabel) {
      progLabel.textContent = `Stage ${activePipelineStep} of 5: ${STEP_TITLES[activePipelineStep]}`;
    }
    if (prevBtn) {
      prevBtn.disabled = activePipelineStep <= 1;
      prevBtn.innerHTML = activePipelineStep > 1
        ? `&#8592; Prev: ${STEP_TITLES[activePipelineStep - 1].split(' (')[0]}`
        : `&#8592; Previous Stage`;
    }
    if (nextBtn) {
      nextBtn.disabled = activePipelineStep >= 5;
      nextBtn.innerHTML = activePipelineStep < 5
        ? `Next: ${STEP_TITLES[activePipelineStep + 1].split(' (')[0]} &#8594;`
        : `Final Stage Reached &#10003;`;
    }

    // Re-draw canvases when their pane becomes visible
    if (currentResults) {
      if (activePipelineStep === 2 || isShowAllMode) {
        drawPreprocessHistogram(currentResults.stage_meta?.preprocess);
      }
      if ((activePipelineStep === 4 || isShowAllMode) && currentResults.kernels?.length) {
        const k = currentResults.kernels[selectedKernelIndex] || currentResults.kernels[0];
        drawFeatureZScoreCanvas(k);
      }
    }
  }

  function selectKernel(index) {
    if (!currentResults || !currentResults.kernels || currentResults.kernels.length === 0) return;
    const total = currentResults.kernels.length;
    selectedKernelIndex = Math.max(0, Math.min(total - 1, index));
    const k = currentResults.kernels[selectedKernelIndex];

    // Highlight active pill and smoothly scroll it into horizontal view
    document.querySelectorAll('.kernel-pill').forEach((pill, idx) => {
      const isAct = idx === selectedKernelIndex;
      pill.classList.toggle('active', isAct);
      if (isAct && typeof pill.scrollIntoView === 'function') {
        pill.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    });

    // Update quick-nav dropdown and prev/next buttons
    const kDropdown = document.getElementById('kernel-select-dropdown');
    const kPrevBtn = document.getElementById('kernel-nav-prev');
    const kNextBtn = document.getElementById('kernel-nav-next');
    if (kDropdown) kDropdown.value = String(selectedKernelIndex);
    if (kPrevBtn) kPrevBtn.disabled = selectedKernelIndex <= 0;
    if (kNextBtn) kNextBtn.disabled = selectedKernelIndex >= total - 1;

    // Highlight active card in #kernels-grid
    document.querySelectorAll('#kernels-grid .kernel-card').forEach((card, idx) => {
      card.classList.toggle('selected-card', idx === selectedKernelIndex);
    });

    // Highlight active row in #feature-table
    document.querySelectorAll('#feature-table tbody tr').forEach((row, idx) => {
      row.classList.toggle('selected-row', idx === selectedKernelIndex);
    });

    // Synchronize mobile pagers (Segmented Kernels Overview & GLCM/HSV Matrix)
    pagerIndex = selectedKernelIndex;
    renderKernelOverviewPager();
    renderPager();

    renderStep3SelectedKernel(k);
    renderStep4SelectedKernel(k);
    renderStep5SelectedKernel(k, currentResults.summary);
  }

  function setupPipelineInspector(data) {
    const images = data.images || {};
    const meta = data.stage_meta || {};
    const kernels = data.kernels || [];

    // 1. Render Step 1 (Input & Standardization)
    const rawImgEl = document.getElementById('step1-raw-img');
    const stdImgEl = document.getElementById('step1-std-img');
    const rawDimEl = document.getElementById('step1-raw-dim');
    const metaGrid1 = document.getElementById('step1-meta-grid');

    if (rawImgEl) rawImgEl.src = images.raw_input || images.original;
    if (stdImgEl) stdImgEl.src = images.original;

    const inMeta = meta.input || {};
    const segMeta = meta.segmentation || {};
    const origW = inMeta.orig_width || 224;
    const origH = inMeta.orig_height || 224;
    if (rawDimEl) rawDimEl.textContent = `${origW} \u00d7 ${origH} px`;

    if (metaGrid1) {
      metaGrid1.innerHTML = `
        <div class="param-item">
          <span class="param-label">Original Resolution</span>
          <span class="param-val">${origW} &times; ${origH} px</span>
          <span class="param-sub">Aspect Ratio ${inMeta.orig_aspect || '1:1'}</span>
        </div>
        <div class="param-item">
          <span class="param-label">Standardized ROI</span>
          <span class="param-val">${inMeta.std_width || 224} &times; ${inMeta.std_height || 224} px</span>
          <span class="param-sub">${inMeta.was_cropped ? 'Auto-centered square ROI crop' : 'Already standardized 224\u00d7224'}</span>
        </div>
        <div class="param-item">
          <span class="param-label">Background Polarity</span>
          <span class="param-val">${segMeta.bg_mode ? segMeta.bg_mode.split(' (')[0] : 'Dark Background'}</span>
          <span class="param-sub">Border Gray: ${segMeta.bg_gray ?? '-'} | Sat: ${segMeta.bg_sat ?? '-'}</span>
        </div>
        <div class="param-item">
          <span class="param-label">Detected Segments</span>
          <span class="param-val">${kernels.length} Object(s)</span>
          <span class="param-sub">${data.summary.sehat} Healthy, ${data.summary.terkontaminasi} Moldy</span>
        </div>
      `;
    }

    // 2. Render Step 2 (Preprocessing)
    const s2Rgb = document.getElementById('step2-rgb-img');
    const s2Gray = document.getElementById('step2-gray-img');
    const s2Blur = document.getElementById('step2-blur-img');
    const s2Stats = document.getElementById('step2-stats-grid');

    if (s2Rgb) s2Rgb.src = images.original;
    if (s2Gray) s2Gray.src = images.gray;
    if (s2Blur) s2Blur.src = images.blur;

    const prepStats = meta.preprocess || {};
    const stdDiff = (prepStats.gray_std && prepStats.blur_std)
      ? Math.max(0, (prepStats.gray_std - prepStats.blur_std)).toFixed(2)
      : '0.45';
    if (s2Stats) {
      s2Stats.innerHTML = `
        <div class="param-item">
          <span class="param-label">Grayscale Mean (&mu;)</span>
          <span class="param-val">${prepStats.gray_mean ?? '-'}</span>
          <span class="param-sub">0 (Black) to 255 (White)</span>
        </div>
        <div class="param-item">
          <span class="param-label">Grayscale Std (&sigma;)</span>
          <span class="param-val">${prepStats.gray_std ?? '-'}</span>
          <span class="param-sub">Raw intensity dispersion</span>
        </div>
        <div class="param-item">
          <span class="param-label">Gaussian Blur (5&times;5)</span>
          <span class="param-val">&sigma; = ${prepStats.blur_std ?? '-'}</span>
          <span class="param-sub">Noise variance drop: -${stdDiff}</span>
        </div>
        <div class="param-item">
          <span class="param-label">Feature Patch Size</span>
          <span class="param-val">128 &times; 128 px</span>
          <span class="param-sub">Bicubic + 3&times;3 Gaussian</span>
        </div>
      `;
    }
    drawPreprocessHistogram(prepStats);

    // 3. Render Step 3A (Global Otsu & Watershed Segmentation)
    const s3Otsu = document.getElementById('step3-otsu-img');
    const s3Overlay = document.getElementById('step3-overlay-img');
    const s3OtsuTag = document.getElementById('step3-otsu-tag');
    const s3RoiCount = document.getElementById('step3-roi-count');
    const s3Stats = document.getElementById('step3-seg-stats');

    if (s3Otsu) s3Otsu.src = images.otsu_mask;
    if (s3Overlay) s3Overlay.src = images.seg_overlay || images.annotated;
    if (s3OtsuTag) s3OtsuTag.textContent = `Otsu Threshold T = ${segMeta.otsu_threshold ?? '-'}`;
    if (s3RoiCount) s3RoiCount.textContent = `${kernels.length} Segmented ROI(s)`;

    if (s3Stats) {
      s3Stats.innerHTML = `
        <div class="param-item">
          <span class="param-label">Otsu Threshold (T*)</span>
          <span class="param-val">${segMeta.otsu_threshold ?? '-'} / 255</span>
          <span class="param-sub">Maximizes inter-class variance &sigma;&sup2;<sub>B</sub></span>
        </div>
        <div class="param-item">
          <span class="param-label">Polarity Mode</span>
          <span class="param-val">${segMeta.bg_mode || 'Standard Polarity'}</span>
          <span class="param-sub">Cool cloth glare suppression active</span>
        </div>
        <div class="param-item">
          <span class="param-label">Morphological Filter</span>
          <span class="param-val">3&times;3 Ellipse Open</span>
          <span class="param-sub">+ Binary Hole Filling</span>
        </div>
        <div class="param-item">
          <span class="param-label">Watershed Splitting</span>
          <span class="param-val">${segMeta.raw_components ?? kernels.length} Blob(s) &rarr; ${kernels.length} ROI(s)</span>
          <span class="param-sub">CLAHE + Distance Transform</span>
        </div>
      `;
    }

    // 4. Populate Kernel Selector Pills & Quick-Nav Dropdown
    const pillsContainer = document.getElementById('kernel-selector-pills');
    const kDropdown = document.getElementById('kernel-select-dropdown');
    if (pillsContainer) {
      pillsContainer.innerHTML = '';
      kernels.forEach((k, idx) => {
        const st = getKernelState(k);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `kernel-pill ${idx === 0 ? 'active' : ''}`;
        btn.innerHTML = `
          <img class="kernel-pill-thumb" src="${k.crop_b64}" alt="Kernel #${k.kernel_id}">
          <span>Kernel #${k.kernel_id}</span>
          <span class="kernel-pill-dot ${st.stateClass}" title="${st.label}"></span>
        `;
        btn.addEventListener('click', () => selectKernel(idx));
        pillsContainer.appendChild(btn);
      });
    }
    if (kDropdown) {
      kDropdown.innerHTML = '';
      kernels.forEach((k, idx) => {
        const st = getKernelState(k);
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = `Kernel #${k.kernel_id} \u2014 ${st.label} (${idx + 1}/${kernels.length})`;
        kDropdown.appendChild(opt);
      });
    }

    // 5. Render Step 5 Annotated Image
    const s5Ann = document.getElementById('step5-annotated-img');
    if (s5Ann) s5Ann.src = images.annotated;

    // Select first kernel by default to populate Steps 3B, 4, and 5
    selectKernel(0);
    switchPipelineStep(activePipelineStep);
  }

  function drawPreprocessHistogram(prepStats) {
    const canvas = document.getElementById('step2-hist-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const hGray = prepStats?.hist_gray || [];
    const hBlur = prepStats?.hist_blur || [];
    if (hGray.length === 0) return;

    const padL = 36, padR = 14, padT = 24, padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const maxVal = Math.max(...hGray, ...hBlur, 1);
    const bins = hGray.length;
    const binW = plotW / bins;

    // Grid lines
    ctx.strokeStyle = '#D9F0E8';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }

    // Bars for Raw Grayscale
    for (let i = 0; i < bins; i++) {
      const bh = (hGray[i] / maxVal) * plotH;
      const x = padL + i * binW + 1;
      const y = padT + plotH - bh;
      ctx.fillStyle = 'rgba(11, 42, 107, 0.30)';
      ctx.fillRect(x, y, Math.max(2, binW - 2), bh);
    }

    // Smooth curve for Gaussian Blurred histogram
    if (hBlur.length === bins) {
      ctx.strokeStyle = '#0FA9A3';
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i < bins; i++) {
        const bh = (hBlur[i] / maxVal) * plotH;
        const cx = padL + (i + 0.5) * binW;
        const cy = padT + plotH - bh;
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = '#2F4577';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('0 (Dark)', padL, H - 8);
    ctx.fillText('128 (Mid)', padL + plotW * 0.45, H - 8);
    ctx.fillText('255 (Bright)', W - padR - 72, H - 8);

    // Legend
    ctx.fillStyle = 'rgba(11, 42, 107, 0.45)';
    ctx.fillRect(padL, 6, 12, 10);
    ctx.fillStyle = '#0B2A6B';
    ctx.fillText('Grayscale Histogram', padL + 17, 15);

    ctx.strokeStyle = '#0FA9A3';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(padL + 165, 11);
    ctx.lineTo(padL + 180, 11);
    ctx.stroke();
    ctx.fillStyle = '#0B7F8C';
    ctx.fillText('After 5x5 Gaussian Blur', padL + 185, 15);
  }

  function renderStep3SelectedKernel(k) {
    const panel = document.getElementById('step3-selected-kernel-panel');
    if (!panel || !k) return;
    const st = getKernelState(k);
    const bbox = k.bbox || [0, 0, 0, 0];

    let swatchesHtml = '';
    if (k.cluster_centers && k.cluster_centers.length > 0) {
      swatchesHtml = `<div class="cluster-swatches">` + k.cluster_centers.map((c, idx) => {
        const cssColor = opencvHsvToCssRgb(c[0], c[1], c[2]);
        return `<span class="cluster-swatch-chip">
          <span class="swatch-box" style="background:${cssColor};"></span>
          C${idx + 1}: H=${c[0]} S=${c[1]} V=${c[2]}
        </span>`;
      }).join('') + `</div>`;
    }

    panel.innerHTML = `
      <div class="step-compare-grid three-col" style="margin-bottom:0.5rem;">
        <div class="visual-box">
          <div class="visual-box-head">
            <span>Kernel #${k.kernel_id} Crop ROI</span>
            <span class="badge ${st.stateClass}">${st.label}</span>
          </div>
          <div class="visual-img-frame" style="max-height:210px;">
            <img src="${k.crop_b64}" alt="Kernel #${k.kernel_id} Crop">
          </div>
          <p class="visual-caption">BBox: <code>[x=${bbox[0]}, y=${bbox[1]}, w=${bbox[2]}, h=${bbox[3]}]</code> &bull; Area: <strong>${k.area_piksel ?? '-'} px</strong></p>
        </div>

        <div class="visual-box">
          <div class="visual-box-head">
            <span>HSV K-Means Quantization</span>
            <span class="mono-tag">K = 3 Clusters</span>
          </div>
          <div class="visual-img-frame" style="max-height:210px;">
            <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Color Quantization">
          </div>
          ${swatchesHtml || '<p class="visual-caption">Pixels grouped into 3 color clusters in HSV space.</p>'}
        </div>

        <div class="visual-box">
          <div class="visual-box-head">
            <span>Isolated Mold Lesion Mask</span>
            <span class="mono-tag">${st.stateClass === 'notcorn' ? 'Excluded' : k.mold_ratio + '% Area'}</span>
          </div>
          <div class="visual-img-frame" style="max-height:210px;">
            <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Lesion Binary Mask">
          </div>
          <p class="visual-caption">Cluster with minimum health score (<code>1.5&times;S + 0.8&times;V</code>) isolated as fungal lesion mask.</p>
        </div>
      </div>
    `;
  }

  function renderStep4SelectedKernel(k) {
    const badgeEl = document.getElementById('step4-kernel-badge');
    const glcmContainer = document.getElementById('step4-glcm-bars');
    const hsvContainer = document.getElementById('step4-hsv-panel');
    if (!k) return;

    const st = getKernelState(k);
    if (badgeEl) badgeEl.textContent = `Kernel #${k.kernel_id} (${st.label})`;

    const f = k.features || {};
    const sf = k.scaled_features || {};

    const glcmSpecs = [
      {
        key: 'contrast',
        name: 'Contrast (Local Intensity Variation)',
        val: f.contrast,
        z: sf.contrast,
        pct: Math.min(100, Math.max(4, (Number(f.contrast) / 60) * 100)),
        desc: 'Measures local intensity variation. Moldy/damaged kernels exhibit higher contrast due to rough fungal lesions.',
      },
      {
        key: 'correlation',
        name: 'Correlation (Linear Gray-Level Dependency)',
        val: f.correlation,
        z: sf.correlation,
        pct: Math.min(100, Math.max(4, Number(f.correlation) * 100)),
        desc: 'Measures linear dependency of neighboring pixels (0 to 1). Smooth pericarp surfaces maintain high correlation.',
      },
      {
        key: 'energy',
        name: 'Energy / Angular Second Moment (Uniformity)',
        val: f.energy,
        z: sf.energy,
        pct: Math.min(100, Math.max(4, (Number(f.energy) / 0.35) * 100)),
        desc: 'Sum of squared GLCM elements. Higher energy indicates uniform, homogeneous surface texture.',
      },
      {
        key: 'homogeneity',
        name: 'Homogeneity (Inverse Difference Moment)',
        val: f.homogeneity,
        z: sf.homogeneity,
        pct: Math.min(100, Math.max(4, Number(f.homogeneity) * 100)),
        desc: 'Measures closeness of GLCM distribution to the diagonal. Drops when fungal spores disrupt surface smoothness.',
      },
    ];

    if (glcmContainer) {
      glcmContainer.innerHTML = glcmSpecs.map((item) => `
        <div class="feat-bar-item">
          <div class="feat-bar-top">
            <span>${item.name}</span>
            <span class="feat-bar-vals">${item.val} ${item.z !== undefined ? `(Z = ${item.z > 0 ? '+' : ''}${item.z})` : ''}</span>
          </div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${item.pct.toFixed(1)}%;"></div>
          </div>
          <div class="feat-bar-desc">${item.desc}</div>
        </div>
      `).join('');
    }

    if (hsvContainer) {
      const liveRgb = opencvHsvToCssRgb(f.hue, f.saturation, f.value);
      const huePct = Math.min(100, Math.max(2, (Number(f.hue) / 179) * 100));
      const satPct = Math.min(100, Math.max(2, (Number(f.saturation) / 255) * 100));
      const valPct = Math.min(100, Math.max(2, (Number(f.value) / 255) * 100));

      hsvContainer.innerHTML = `
        <div class="hsv-swatch-row">
          <div class="hsv-live-swatch" style="background:${liveRgb};" title="Mean Kernel Color"></div>
          <div>
            <div style="font-size:0.82rem; font-weight:600; color:var(--navy);">Mean Foreground HSV Color Swatch</div>
            <div style="font-family:var(--font-mono); font-size:0.74rem; color:var(--teal-dark);">
              H = ${f.hue} (OpenCV 0–179) &bull; S = ${f.saturation} (0–255) &bull; V = ${f.value} (0–255)
            </div>
            <div style="font-size:0.69rem; color:var(--text-secondary);">
              Valid Corn Hue Gate: <code>12.0 &le; Hue &le; 41.0</code> (Golden-yellow to olive-brown)
            </div>
          </div>
        </div>
        <div class="feat-bar-item">
          <div class="feat-bar-top">
            <span>Hue (Dominant Wavelength)</span>
            <span class="feat-bar-vals">${f.hue} ${sf.hue !== undefined ? `(Z = ${sf.hue > 0 ? '+' : ''}${sf.hue})` : ''}</span>
          </div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${huePct.toFixed(1)}%; background:linear-gradient(90deg, #EAB308, #10A54A);"></div>
          </div>
        </div>
        <div class="feat-bar-item">
          <div class="feat-bar-top">
            <span>Saturation (Chroma Purity)</span>
            <span class="feat-bar-vals">${f.saturation} ${sf.saturation !== undefined ? `(Z = ${sf.saturation > 0 ? '+' : ''}${sf.saturation})` : ''}</span>
          </div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${satPct.toFixed(1)}%;"></div>
          </div>
        </div>
        <div class="feat-bar-item">
          <div class="feat-bar-top">
            <span>Value (Luminance Brightness)</span>
            <span class="feat-bar-vals">${f.value} ${sf.value !== undefined ? `(Z = ${sf.value > 0 ? '+' : ''}${sf.value})` : ''}</span>
          </div>
          <div class="feat-bar-track">
            <div class="feat-bar-fill" style="width:${valPct.toFixed(1)}%;"></div>
          </div>
        </div>
      `;
    }

    drawFeatureZScoreCanvas(k);
  }

  function drawFeatureZScoreCanvas(k) {
    const canvas = document.getElementById('step4-feature-canvas');
    if (!canvas || !k) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const sf = k.scaled_features || {
      contrast: 0, correlation: 0, energy: 0, homogeneity: 0, hue: 0, saturation: 0, value: 0,
    };
    const keys = ['contrast', 'correlation', 'energy', 'homogeneity', 'hue', 'saturation', 'value'];
    const labels = ['Contrast', 'Correl', 'Energy', 'Homog', 'Hue', 'Sat', 'Value'];

    const padL = 38, padR = 14, padT = 24, padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const midY = padT + plotH / 2;

    // Zero baseline
    ctx.strokeStyle = '#9BD8F0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, midY);
    ctx.lineTo(W - padR, midY);
    ctx.stroke();

    ctx.fillStyle = '#0B2A6B';
    ctx.font = '600 11px Poppins, sans-serif';
    ctx.fillText(`Standardized Feature Vector X_scaled (Z-Scores for Kernel #${k.kernel_id})`, padL, 15);

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#5B6B8C';
    ctx.fillText('+3\u03c3', 8, padT + 8);
    ctx.fillText(' 0', 12, midY + 4);
    ctx.fillText('-3\u03c3', 8, padT + plotH);

    const colW = plotW / keys.length;
    keys.forEach((key, idx) => {
      const z = Math.max(-3.2, Math.min(3.2, Number(sf[key] || 0)));
      const barH = (Math.abs(z) / 3.2) * (plotH / 2);
      const x = padL + idx * colW + colW * 0.18;
      const bw = colW * 0.64;
      const y = z >= 0 ? midY - barH : midY;

      ctx.fillStyle = z >= 0 ? '#0FA9A3' : '#0B2A6B';
      ctx.fillRect(x, y, bw, Math.max(2, barH));

      ctx.fillStyle = '#1F2A44';
      ctx.fillText(labels[idx], padL + idx * colW + 6, H - 8);
    });
  }

  function renderStep5SelectedKernel(k, summary) {
    const container = document.getElementById('step5-decision-inspector');
    const batchBox = document.getElementById('step5-batch-summary-box');
    if (!container || !k) return;

    const st = getKernelState(k);
    const f = k.features || {};
    const hueOk = f.hue >= 12.0 && f.hue <= 41.0;
    const morphOk = (k.area_piksel || 500) <= 10500 && (k.aspect_ratio || 1.2) <= 2.35 && (k.solidity || 0.9) >= 0.58;
    const od = k.outlier_distance;
    const ot = k.outlier_threshold;
    const mahaOk = k.is_corn && (od === null || ot === null || od <= ot);

    const dec = k.decision_score;
    // Map decision score [-3, +3] to percentage [5%, 95%] where 50% is hyperplane f(x)=0
    const markerPct = dec !== null && dec !== undefined
      ? Math.max(5, Math.min(95, 50 + (Number(dec) / 3.0) * 45))
      : 50;

    const probHealthy = k.probabilities?.sehat ?? 0;
    const probMoldy = k.probabilities?.terkontaminasi ?? 0;

    container.innerHTML = `
      <div class="sub-panel">
        <div class="sub-panel-head">
          <h4 class="sub-panel-title">5A. Multi-Layer Novelty Gate (<code>check_is_corn</code>)</h4>
          <span class="badge ${st.stateClass}">Kernel #${k.kernel_id}: ${st.label}</span>
        </div>
        <div class="gate-checklist">
          <div class="gate-item">
            <div>
              <strong>Layer 1: Color &amp; Warmth Constraints</strong>
              <div style="font-size:0.7rem; color:var(--text-secondary);">Hue = ${f.hue} (valid [12, 41]), Sat = ${f.saturation}, Warmth R &gt; B</div>
            </div>
            <span class="gate-status ${hueOk ? 'pass' : 'fail'}">${hueOk ? 'PASS' : 'REJECT'}</span>
          </div>
          <div class="gate-item">
            <div>
              <strong>Layer 2: Morphological Geometry</strong>
              <div style="font-size:0.7rem; color:var(--text-secondary);">Area = ${k.area_piksel ?? '-'} px, Aspect = ${k.aspect_ratio ?? '-'}:1, Solidity = ${k.solidity ?? '-'}</div>
            </div>
            <span class="gate-status ${morphOk ? 'pass' : 'fail'}">${morphOk ? 'PASS' : 'REJECT'}</span>
          </div>
          <div class="gate-item">
            <div>
              <strong>Layer 3: 7D Subcluster Mahalanobis Distance</strong>
              <div style="font-size:0.7rem; color:var(--text-secondary);">
                Distance d<sub>M</sub> = <code>${od ?? '-'}</code> vs Threshold &tau; = <code>${ot ?? '-'}</code>
                ${k.outlier_nearest_class ? `(Nearest: ${formatStatus(k.outlier_nearest_class)})` : ''}
              </div>
            </div>
            <span class="gate-status ${mahaOk ? 'pass' : 'fail'}">${mahaOk ? 'VALID CORN' : 'OUTLIER'}</span>
          </div>
        </div>
      </div>

      <div class="sub-panel">
        <div class="sub-panel-head">
          <h4 class="sub-panel-title">5B. RBF-SVM Hyperplane Decision Function</h4>
          <span class="mono-tag">${dec !== null && dec !== undefined ? `f(x) = ${dec > 0 ? '+' : ''}${dec}` : 'Bypassed (Not Corn)'}</span>
        </div>
        ${k.prediction === 'bukan_jagung' ? `
          <div class="theory-box" style="border-left-color:var(--notcorn-gray);">
            <strong>Rejected by Novelty Gate:</strong>
            <span>Kernel #${k.kernel_id} was rejected as a non-corn object before SVM classification and is excluded from the batch contamination rate.</span>
          </div>
        ` : `
          <div class="hyperplane-meter">
            <div class="hyperplane-scale">
              <span style="color:var(--healthy-green);">&#8592; Healthy Class (f(x) &lt; 0)</span>
              <span>Hyperplane (0.0)</span>
              <span style="color:var(--mold-red);">Mold Class (f(x) &gt; 0) &#8594;</span>
            </div>
            <div class="hyperplane-track">
              <div class="hyperplane-zero" title="Decision Boundary f(x) = 0"></div>
              <div class="hyperplane-marker" style="left:${markerPct.toFixed(1)}%;" title="f(x) = ${dec}"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:0.75rem; margin-top:0.4rem;">
              <span>P(Healthy): <strong>${probHealthy}%</strong></span>
              <span>Confidence: <strong>${k.confidence}%</strong></span>
              <span>P(Mold): <strong>${probMoldy}%</strong></span>
            </div>
          </div>
          <div class="theory-box">
            <strong>Platt Sigmoid Calibration:</strong>
            <span>RBF-SVM maps the 7D scaled vector into a high-dimensional kernel space and computes signed distance <code>f(x) = ${dec}</code> from the separating hyperplane, converted to posterior probability via <code>P = 1 / (1 + exp(-f(x)))</code>.</span>
          </div>
        `}
      </div>
    `;

    if (batchBox && summary) {
      const totalJagung = (summary.sehat || 0) + (summary.terkontaminasi || 0);
      batchBox.innerHTML = `
        <div class="visual-box-head">
          <span>Batch Quality Grading Summary</span>
          <span class="mono-tag">Contamination: ${summary.persen_kontaminasi}%</span>
        </div>
        <div class="step-params-grid compact" style="margin-top:0.25rem;">
          <div class="param-item">
            <span class="param-label">Valid Corn Kernels</span>
            <span class="param-val">${totalJagung} Kernel(s)</span>
            <span class="param-sub">Excluded Non-Corn: ${summary.bukan_jagung || 0}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Class Breakdown</span>
            <span class="param-val" style="color:var(--healthy-green);">${summary.sehat} Healthy</span> /
            <span class="param-val" style="color:var(--mold-red);">${summary.terkontaminasi} Mold</span>
            <span class="param-sub">Ratio: ${summary.persen_kontaminasi}% infected</span>
          </div>
        </div>
        <div class="theory-box" style="margin-top:auto;">
          <strong>Final Quality Verdict:</strong>
          <span>${summary.status_mutu}</span>
          <div style="margin-top:0.35rem; font-size:0.72rem; color:var(--text-secondary);">
            Grading Thresholds: <strong>0%</strong> = Excellent (Seed/Consumption) &bull; <strong>&le; 15%</strong> = Standard (Sorting Required) &bull; <strong>&gt; 15%</strong> = Poor Quality (Unfit).
          </div>
        </div>
      `;
    }
  }

  // Bind Kernel Quick-Nav (Prev / Dropdown / Next)
  const kNavPrev = document.getElementById('kernel-nav-prev');
  const kNavNext = document.getElementById('kernel-nav-next');
  const kSelectDropdown = document.getElementById('kernel-select-dropdown');
  if (kNavPrev) {
    kNavPrev.addEventListener('click', () => {
      if (selectedKernelIndex > 0) selectKernel(selectedKernelIndex - 1);
    });
  }
  if (kNavNext) {
    kNavNext.addEventListener('click', () => {
      if (currentResults?.kernels && selectedKernelIndex < currentResults.kernels.length - 1) {
        selectKernel(selectedKernelIndex + 1);
      }
    });
  }
  if (kSelectDropdown) {
    kSelectDropdown.addEventListener('change', (e) => {
      selectKernel(Number(e.target.value));
    });
  }

  // Bind Stepper Tabs, Prev/Next Buttons, and Mode Switcher
  document.querySelectorAll('.stepper-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      switchPipelineStep(Number(tab.dataset.step));
    });
  });

  const btnStepPrev = document.getElementById('btn-step-prev');
  const btnStepNext = document.getElementById('btn-step-next');
  if (btnStepPrev) {
    btnStepPrev.addEventListener('click', () => {
      if (activePipelineStep > 1) switchPipelineStep(activePipelineStep - 1);
    });
  }
  if (btnStepNext) {
    btnStepNext.addEventListener('click', () => {
      if (activePipelineStep < 5) switchPipelineStep(activePipelineStep + 1);
    });
  }

  const btnModeStepper = document.getElementById('btn-mode-stepper');
  const btnModeAll = document.getElementById('btn-mode-all');
  const inspectorCard = document.querySelector('.pipeline-inspector-card');
  const stepperFooter = document.getElementById('stepper-footer');

  if (btnModeStepper && btnModeAll && inspectorCard) {
    btnModeStepper.addEventListener('click', () => {
      isShowAllMode = false;
      inspectorCard.classList.remove('show-all-steps');
      btnModeStepper.classList.add('active-mode');
      btnModeAll.classList.remove('active-mode');
      if (stepperFooter) stepperFooter.style.display = 'flex';
      switchPipelineStep(activePipelineStep);
    });
    btnModeAll.addEventListener('click', () => {
      isShowAllMode = true;
      inspectorCard.classList.add('show-all-steps');
      btnModeAll.classList.add('active-mode');
      btnModeStepper.classList.remove('active-mode');
      if (stepperFooter) stepperFooter.style.display = 'none';
      switchPipelineStep(activePipelineStep);
    });
  }

  function setupStages(images) {
    const stageImg = document.getElementById('stage-image');
    const stageDesc = document.getElementById('stage-desc');
    const stageButtons = document.querySelectorAll('.stage-btn');

    const stageData = {
      annotated: {
        img: images.annotated,
        desc: 'Final classification result of the Support Vector Machine (SVM): green bounding boxes indicate healthy kernels, red indicates mold-contaminated kernels, and gray indicates objects detected as NOT corn kernels (excluded from the quality assessment).',
      },
      original: {
        img: images.original,
        desc: 'Original high-resolution corn kernel image before preprocessing.',
      },
      'gray-blur': {
        img: images.blur,
        desc: 'Preprocessing Step 1: Conversion of the image to grayscale and noise reduction using Gaussian Blur.',
      },
      otsu: {
        img: images.otsu_mask,
        desc: 'Segmentation Step 2A: Separation of kernel objects from the background using Otsu Thresholding and an elliptical morphological filter.',
      },
    };

    // Set default to annotated
    stageImg.src = stageData['annotated'].img;
    stageDesc.textContent = stageData['annotated'].desc;

    stageButtons.forEach((btn) => {
      btn.onclick = () => {
        stageButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const stageKey = btn.dataset.stage;
        if (stageData[stageKey]) {
          stageImg.src = stageData[stageKey].img;
          stageDesc.textContent = stageData[stageKey].desc;
        }
      };
    });
  }

  function renderKernelCards(kernels) {
    const container = document.getElementById('kernels-grid');
    container.innerHTML = '';

    kernels.forEach((k, idx) => {
      // BARU -- pakai helper getKernelState() alih-alih cuma cek k.is_moldy,
      // supaya ada cabang ketiga untuk "bukan_jagung".
      const state = getKernelState(k);

      const card = document.createElement('div');
      // style.css sekarang sudah punya rule .kernel-card.notcorn (border atas
      // abu-abu), jadi cukup pasang class-nya saja -- sama seperti healthy/moldy.
      card.className = `kernel-card selectable-kernel ${state.stateClass} ${idx === selectedKernelIndex ? 'selected-card' : ''}`;
      card.addEventListener('click', () => selectKernel(idx));

      // BARU -- confidence bisa null untuk bukan_jagung, jadi tampilkan "-"
      // alih-alih literal teks "null%".
      const confidenceText = k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : null;

      // BARU -- untuk bukan_jagung, "Area Jamur X%" tidak relevan (itu cuma
      // hasil clustering warna, bukan indikasi jamur beneran) -- diganti
      // dengan info jarak outlier supaya tetap informatif & jujur secara metodologis.
      const thirdThumbLabel = state.stateClass === 'notcorn' ? `Not Corn (distance ${k.outlier_distance ?? '-'} vs threshold ${k.outlier_threshold ?? '-'})` : `Mold Area (${k.mold_ratio}%)`;

      card.innerHTML = `
                <div class="kernel-header">
                    <span class="kernel-title">Kernel #${k.kernel_id}</span>
                    <span class="badge ${state.stateClass}">${state.label}</span>
                </div>
                <div class="kernel-visuals">
                    <div class="thumb-box">
                        <img src="${k.crop_b64}" alt="Original Crop">
                        <span>Kernel Crop</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Cluster">
                        <span>K-Means (Color)</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Mask">
                        <span>${thirdThumbLabel}</span>
                    </div>
                </div>
                <div class="kernel-features-mini">
                    <div>Contrast: <span class="feature-val">${k.features.contrast}</span></div>
                    <div>Homogeneity: <span class="feature-val">${k.features.homogeneity}</span></div>
                    <div>Correlation: <span class="feature-val">${k.features.correlation}</span></div>
                    <div>Energy: <span class="feature-val">${k.features.energy}</span></div>
                    <div>Hue: <span class="feature-val">${k.features.hue}</span></div>
                    <div>Sat: <span class="feature-val">${k.features.saturation}</span></div>
                </div>
                ${confidenceText === null ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence not applicable -- object rejected before the SVM classification stage (see the distance details above)</div>` : ''}
            `;
      container.appendChild(card);
    });
  }

  // Peta nilai "prediction" dari backend -> label Inggris untuk tampilan.
  // KEY (sehat / terkontaminasi / bukan_jagung) JANGAN diubah: itu nilai yang dikirim
  // pipeline.py dan dipakai logika lain. Yang diterjemahkan hanya teks yang tampil.
  const PREDICTION_LABELS = {
    sehat: 'Healthy',
    terkontaminasi: 'Contaminated',
    bukan_jagung: 'Not Corn',
  };
  function formatStatus(prediction) {
    return PREDICTION_LABELS[prediction] || String(prediction);
  }

  // confidence null ditampilkan "-", bukan "null%"
  function formatConfidence(k) {
    return k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : '-';
  }

  // --- State pager (tampilan mobile: 1 biji per halaman) ---
  let pagerKernels = [];
  let pagerIndex = 0;
  const pagerCard = document.getElementById('pager-card');
  const pagerInfo = document.getElementById('pager-info');
  const pagerPrev = document.getElementById('pager-prev');
  const pagerNext = document.getElementById('pager-next');

  // --- Mobile Pager for All Segmented Kernels Overview ---
  const kernelPagerCard = document.getElementById('kernel-overview-pager-card');
  const kernelPagerInfo = document.getElementById('kernel-pager-info');
  const kernelPagerPrev = document.getElementById('kernel-pager-prev');
  const kernelPagerNext = document.getElementById('kernel-pager-next');

  function renderKernelOverviewPager() {
    if (!kernelPagerCard || !kernelPagerInfo || !kernelPagerPrev || !kernelPagerNext) return;
    const kernels = currentResults?.kernels || [];
    const total = kernels.length;
    if (total === 0) {
      kernelPagerCard.innerHTML = `<div class="pager-empty">No kernels detected yet.</div>`;
      kernelPagerInfo.textContent = '0 of 0';
      kernelPagerPrev.disabled = true;
      kernelPagerNext.disabled = true;
      return;
    }

    const idx = Math.max(0, Math.min(total - 1, selectedKernelIndex));
    const k = kernels[idx];
    const state = getKernelState(k);
    const confidenceText = k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : null;
    const thirdThumbLabel = state.stateClass === 'notcorn'
      ? `Not Corn (distance ${k.outlier_distance ?? '-'} vs threshold ${k.outlier_threshold ?? '-'})`
      : `Mold Area (${k.mold_ratio}%)`;

    kernelPagerCard.innerHTML = `
      <div class="kernel-card ${state.stateClass} selected-card">
        <div class="kernel-header">
          <span class="kernel-title">Kernel #${k.kernel_id}</span>
          <span class="badge ${state.stateClass}">${state.label}</span>
        </div>
        <div class="kernel-visuals">
          <div class="thumb-box">
            <img src="${k.crop_b64}" alt="Original Crop">
            <span>Kernel Crop</span>
          </div>
          <div class="thumb-box">
            <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Cluster">
            <span>K-Means (Color)</span>
          </div>
          <div class="thumb-box">
            <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Mask">
            <span>${thirdThumbLabel}</span>
          </div>
        </div>
        <div class="kernel-features-mini">
          <div>Contrast: <span class="feature-val">${k.features.contrast}</span></div>
          <div>Homogeneity: <span class="feature-val">${k.features.homogeneity}</span></div>
          <div>Correlation: <span class="feature-val">${k.features.correlation}</span></div>
          <div>Energy: <span class="feature-val">${k.features.energy}</span></div>
          <div>Hue: <span class="feature-val">${k.features.hue}</span></div>
          <div>Sat: <span class="feature-val">${k.features.saturation}</span></div>
        </div>
        ${confidenceText === null ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence not applicable -- object rejected before the SVM classification stage (see the distance details above)</div>` : ''}
      </div>
    `;

    kernelPagerInfo.textContent = `${idx + 1} of ${total}`;
    kernelPagerPrev.disabled = idx === 0;
    kernelPagerNext.disabled = idx === total - 1;
  }

  if (kernelPagerPrev) {
    kernelPagerPrev.addEventListener('click', () => {
      if (selectedKernelIndex > 0) {
        selectKernel(selectedKernelIndex - 1);
      }
    });
  }
  if (kernelPagerNext) {
    kernelPagerNext.addEventListener('click', () => {
      const total = currentResults?.kernels?.length || 0;
      if (selectedKernelIndex < total - 1) {
        selectKernel(selectedKernelIndex + 1);
      }
    });
  }

  function renderPager() {
    const total = pagerKernels.length;
    if (total === 0) {
      pagerCard.innerHTML = `<div class="pager-empty">No kernels detected yet.</div>`;
      pagerInfo.textContent = '0 of 0';
      pagerPrev.disabled = true;
      pagerNext.disabled = true;
      return;
    }

    const k = pagerKernels[pagerIndex];
    const state = getKernelState(k);
    const rows = [
      ['SVM Status', `<span class="st-${state.stateClass}">${formatStatus(k.prediction)}</span>`],
      ['Conf', formatConfidence(k)],
      ['% Mold', `${k.mold_ratio}%`],
      ['Contrast', k.features.contrast],
      ['Correlation', k.features.correlation],
      ['Energy', k.features.energy],
      ['Homogeneity', k.features.homogeneity],
      ['Hue', k.features.hue],
      ['Saturation', k.features.saturation],
      ['Value', k.features.value],
    ];

    pagerCard.innerHTML =
      `<div class="pager-head">
                <img class="pager-thumb" src="${k.crop_b64}" alt="Kernel crop #${k.kernel_id}">
                <span class="pager-id">#${k.kernel_id}</span>
            </div>` + rows.map((r) => `<div class="pager-row"><span class="pager-label">${r[0]}</span><span class="pager-value">${r[1]}</span></div>`).join('');

    pagerInfo.textContent = `${pagerIndex + 1} of ${total}`;
    pagerPrev.disabled = pagerIndex === 0;
    pagerNext.disabled = pagerIndex === total - 1;
  }

  pagerPrev.addEventListener('click', () => {
    if (pagerIndex > 0) {
      selectKernel(pagerIndex - 1);
    }
  });
  pagerNext.addEventListener('click', () => {
    if (pagerIndex < pagerKernels.length - 1) {
      selectKernel(pagerIndex + 1);
    }
  });

  function renderFeatureTable(kernels) {
    const tbody = document.querySelector('#feature-table tbody');
    tbody.innerHTML = '';

    kernels.forEach((k, idx) => {
      const tr = document.createElement('tr');
      if (idx === selectedKernelIndex) tr.classList.add('selected-row');
      tr.addEventListener('click', () => selectKernel(idx));
      const state = getKernelState(k);

      tr.innerHTML = `
                <td>#${k.kernel_id}</td>
                <td class="st-${state.stateClass}">${formatStatus(k.prediction)}</td>
                <td>${formatConfidence(k)}</td>
                <td>${k.mold_ratio}%</td>
                <td>${k.features.contrast}</td>
                <td>${k.features.correlation}</td>
                <td>${k.features.energy}</td>
                <td>${k.features.homogeneity}</td>
                <td>${k.features.hue}</td>
                <td>${k.features.saturation}</td>
                <td>${k.features.value}</td>
            `;
      tbody.appendChild(tr);
    });

    // Isi pager mobile dengan data yang sama, mulai dari biji pertama
    pagerKernels = kernels;
    pagerIndex = 0;
    renderPager();
  }

  // Export CSV
  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!currentResults || !currentResults.kernels) return;

      let csv = 'ID,Status,Confidence(%),MoldArea(%),Contrast,Correlation,Energy,Homogeneity,Hue,Saturation,Value\n';
      currentResults.kernels.forEach((k) => {
        // BARU -- confidence null ditulis string kosong di CSV, bukan literal "null".
        const confVal = k.confidence !== null && k.confidence !== undefined ? k.confidence : '';
        csv += `${k.kernel_id},${formatStatus(k.prediction)},${confVal},${k.mold_ratio},${k.features.contrast},${k.features.correlation},${k.features.energy},${k.features.homogeneity},${k.features.hue},${k.features.saturation},${k.features.value}\n`;
      });

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `corn_disease_features_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ============================
  // BARU -- REALTIME WEBCAM DETECTION
  // ============================
  const webcamVideo = document.getElementById('webcam-video');
  const webcamCanvas = document.getElementById('webcam-canvas');
  const realtimeOutput = document.getElementById('realtime-output');
  const realtimeSummary = document.getElementById('realtime-summary');
  const realtimeStatus = document.getElementById('realtime-status');
  const btnStartWebcam = document.getElementById('btn-start-webcam');
  const btnStopWebcam = document.getElementById('btn-stop-webcam');

  let webcamStream = null; // menyimpan stream kamera aktif, supaya bisa dimatikan nanti
  let realtimeIntervalId = null; // id dari setInterval, supaya bisa dihentikan saat klik stop
  let isProcessingFrame = false; // "kunci" -- cegah kirim frame baru sebelum frame sebelumnya selesai diproses server
  let realtimeHasDetection = false; // flag: apakah ada biji terdeteksi selama sesi realtime ini
  let lastRealtimeFrame = null; // menyimpan frame terakhir (base64) untuk analisis penuh saat kamera dihentikan

  const REALTIME_INTERVAL_MS = 400; // jeda ambil frame: 400ms (~2.5 FPS) untuk deteksi realtime lebih cepat

  async function startWebcam() {
    // Cek dulu apakah browser menyediakan akses kamera
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('The camera can only be used over HTTPS or localhost. Open the app via an https:// address or use one of the solutions in the guide.');
      return;
    }

    try {
      // Minta akses kamera belakang (environment). Browser akan menampilkan dialog izin.
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
    } catch (err) {
      alert('Unable to access the camera: ' + err.message);
      return;
    }

    webcamVideo.srcObject = webcamStream; // sambungkan stream kamera ke elemen <video>

    btnStartWebcam.disabled = true;
    btnStopWebcam.disabled = false;
    realtimeStatus.textContent = 'Camera active, starting detection...';
    realtimeStatus.classList.remove('is-live', 'is-error');
    realtimeHasDetection = false;
    lastRealtimeFrame = null;

    // setInterval memanggil captureAndDetectFrame() berulang-ulang tiap
    // REALTIME_INTERVAL_MS milidetik, selama kamera masih menyala.
    realtimeIntervalId = setInterval(captureAndDetectFrame, REALTIME_INTERVAL_MS);
  }

  // ---- Helper: matikan kamera & hentikan interval (tanpa analisis) ----
  // Dipakai oleh beforeunload dan sebagai bagian dari stopWebcamAndAnalyze().
  function stopWebcamCleanup() {
    if (realtimeIntervalId) {
      clearInterval(realtimeIntervalId);
      realtimeIntervalId = null;
    }
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    webcamVideo.srcObject = null;
    isProcessingFrame = false;
  }

  // ---- Hentikan kamera + jalankan analisis penuh pada frame terakhir ----
  // Dipanggil saat user klik tombol "Hentikan Kamera". Kalau selama sesi
  // realtime ada biji yang terdeteksi, frame terakhir dari video akan
  // diproses oleh pipeline penuh (bukan lightweight) supaya hasilnya
  // bisa ditampilkan lengkap: tahapan preprocessing, kartu per-biji
  // dengan K-Means clustering, dan tabel fitur GLCM & HSV.
  function captureStandardizedSquare(videoEl, targetDim = 224, quality = 0.85) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const side = Math.min(vw, vh);
    const sx = Math.floor((vw - side) / 2);
    const sy = Math.floor((vh - side) / 2);
    webcamCanvas.width = targetDim;
    webcamCanvas.height = targetDim;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(videoEl, sx, sy, side, side, 0, 0, targetDim, targetDim);
    return webcamCanvas.toDataURL('image/jpeg', quality);
  }

  async function stopWebcamAndAnalyze() {
    // 1. Tangkap frame terakhir SEBELUM kamera dimatikan (otomatis center-crop & resize 224x224)
    let finalFrame = null;
    if (webcamVideo.videoWidth) {
      finalFrame = captureStandardizedSquare(webcamVideo, 224, 0.92);
    }
    // Fallback ke frame terakhir yang tersimpan kalau capture gagal
    if (!finalFrame && lastRealtimeFrame) {
      finalFrame = lastRealtimeFrame;
    }

    // 2. Matikan kamera
    stopWebcamCleanup();

    btnStartWebcam.disabled = false;
    btnStopWebcam.disabled = true;
    realtimeStatus.textContent = finalFrame ? 'Camera stopped — processing the last frame...' : 'Camera stopped';
    realtimeStatus.classList.remove('is-live', 'is-error');

    // 3. Kalau ada frame dengan deteksi, jalankan pipeline PENUH
    //    (endpoint /api/detect/frame/full mengembalikan semua data lengkap:
    //    gambar tahapan, crop biji, K-Means, mold mask, fitur GLCM & HSV)
    if (finalFrame) {
      showLoading(true);
      try {
        const res = await fetch('/api/detect/frame/full', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: finalFrame }),
        });
        if (res.ok) {
          const data = await res.json();
          displayResults(data);
          realtimeStatus.textContent = 'Camera stopped — full analysis results are shown below';
        } else {
          realtimeStatus.textContent = 'Camera stopped (failed to analyze the last frame)';
          realtimeStatus.classList.add('is-error');
        }
      } catch (err) {
        console.error('Gagal analisis frame terakhir:', err);
        realtimeStatus.textContent = 'Camera stopped (server connection error)';
        realtimeStatus.classList.add('is-error');
      } finally {
        showLoading(false);
      }
    }

    // Reset untuk sesi kamera berikutnya
    realtimeHasDetection = false;
    lastRealtimeFrame = null;
  }

  async function captureAndDetectFrame() {
    // Kalau frame sebelumnya masih diproses server, jangan kirim frame baru dulu --
    // supaya request tidak menumpuk kalau server ternyata lebih lambat dari interval.
    if (isProcessingFrame) return;
    // Video belum siap (ukuran masih 0) -> tunggu interval berikutnya
    if (!webcamVideo.videoWidth) return;

    // Otomatis center-crop dan resize frame video ke resolusi standar 224x224
    // agar pengiriman data dan pemrosesan realtime jauh lebih cepat dan efisien.
    const frameDataUrl = captureStandardizedSquare(webcamVideo, 224, 0.82);
    lastRealtimeFrame = frameDataUrl; // simpan untuk analisis penuh saat kamera dihentikan

    isProcessingFrame = true;
    try {
      const res = await fetch('/api/detect/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frameDataUrl }),
      });

      if (!res.ok) {
        // Kalau 1 frame gagal diproses, jangan hentikan seluruh sesi realtime --
        // cukup catat di status, lanjut coba lagi di frame berikutnya.
        realtimeStatus.textContent = 'This frame failed to process, retrying...';
        realtimeStatus.classList.add('is-error');
        return;
      }

      const data = await res.json();
      renderRealtimeResult(data);
    } catch (err) {
      realtimeStatus.textContent = 'Server connection problem...';
      realtimeStatus.classList.add('is-error');
    } finally {
      isProcessingFrame = false;
    }
  }

  function renderRealtimeResult(data) {
    // Kalau kamera sudah dihentikan saat request masih jalan, abaikan hasilnya
    if (!webcamStream) return;
    realtimeOutput.src = data.annotated;

    const s = data.summary;
    realtimeStatus.textContent = `Live -- kernels detected: ${s.total_biji}`;
    realtimeStatus.classList.remove('is-error');
    realtimeStatus.classList.add('is-live');

    // Tandai bahwa ada OBJEK APAPUN terdeteksi di sesi ini -- supaya saat kamera
    // dihentikan, kita tahu perlu jalankan analisis penuh pada frame terakhir.
    // Pakai data.kernels.length (bukan s.total_biji) karena total_biji mungkin
    // tidak termasuk objek "bukan jagung", padahal kita tetap ingin menampilkan
    // analisis lengkap meskipun semua objek terdeteksi sebagai bukan jagung.
    if (data.kernels && data.kernels.length > 0) {
      realtimeHasDetection = true;
    }
    const totalJagung = (s.sehat || 0) + (s.terkontaminasi || 0);
    const bukanJagung = s.bukan_jagung || 0;
    let text = '';
    if (totalJagung === 0) {
      text = bukanJagung > 0 ? `Not Corn (${bukanJagung} non-corn object(s) detected).` : 'No corn kernels detected in this frame yet.';
    } else if (bukanJagung > totalJagung) {
      text = `Not Corn -- non-corn objects predominate (${bukanJagung} non-corn vs ${totalJagung} corn-like).`;
    } else {
      text = `${s.sehat} healthy, ${s.terkontaminasi} mold-contaminated (${s.persen_kontaminasi}%)`;
      if (bukanJagung > 0) {
        text += ` -- ignored non-corn objects: ${bukanJagung}.`;
      }
    }
    realtimeSummary.textContent = text;
  }

  btnStartWebcam.addEventListener('click', startWebcam);
  btnStopWebcam.addEventListener('click', stopWebcamAndAnalyze);

  // Kalau user pindah/tutup tab tanpa klik "Hentikan Kamera" dulu, pastikan
  // kamera tetap dimatikan supaya lampu indikator kamera tidak terus menyala.
  // Pakai stopWebcamCleanup (bukan stopWebcamAndAnalyze) karena tab sudah
  // ditutup -- tidak perlu kirim request analisis yang hasilnya tidak akan dilihat.
  window.addEventListener('beforeunload', stopWebcamCleanup);

  loadSamples();
});