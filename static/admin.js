// =====================================================================
// Crowdsourcing Dataset & Continuous Training -- Frontend
// - Notifikasi kontribusi dataset + feedback opsional untuk user biasa
// - Login admin, dashboard kurasi staging, re-training & rollback model
// =====================================================================

(() => {
  const TOKEN_KEY = 'cornAdminToken';
  const LABELS = {
    sehat: 'Healthy',
    terkontaminasi: 'Mold Contaminated',
    campuran: 'Mixed',
  };
  const SOURCE_LABELS = { upload: 'Upload', webcam: 'Camera' };

  const $ = (id) => document.getElementById(id);
  const escapeHtml = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  let adminToken = sessionStorage.getItem(TOKEN_KEY);
  let stagingItems = [];
  let activeFilter = 'all';
  let retrainPollId = null;

  // -------------------------------------------------------------------
  // 1. NOTIFIKASI STAGING UNTUK USER BIASA (dipanggil dari app.js)
  // -------------------------------------------------------------------
  window.renderStagingNotice = function (staging) {
    const box = $('staging-notice');
    if (!box) return;

    if (!staging || !staging.saved) {
      if (staging && staging.reason === 'not_corn') {
        box.className = 'staging-notice muted';
        box.innerHTML = '<strong>Not added to the dataset.</strong> No valid corn kernels were found, so this photo was discarded.';
        box.hidden = false;
      } else if (staging && staging.reason === 'duplicate') {
        box.className = 'staging-notice muted';
        box.innerHTML = 'This exact photo is already in the dataset queue.';
        box.hidden = false;
      } else {
        box.hidden = true;
      }
      return;
    }

    const cat = staging.category;
    const predText = cat === 'campuran' ? 'mixed (healthy + moldy kernels)' : LABELS[cat];
    box.className = 'staging-notice';
    box.innerHTML = `
      <div class="staging-notice-text">
        <strong>Thank you! This photo was added to the dataset review queue.</strong>
        <span>Predicted as <b>${escapeHtml(predText)}</b>. An admin will verify it before it is used to re-train the model.</span>
      </div>
      <div class="staging-feedback" data-id="${escapeHtml(staging.id)}">
        <span>Is this result correct?</span>
        <button type="button" class="btn btn-outline" data-fb="correct">Yes, correct</button>
        <button type="button" class="btn btn-outline" data-fb="sehat">No, all healthy</button>
        <button type="button" class="btn btn-outline" data-fb="terkontaminasi">No, moldy</button>
      </div>`;
    box.hidden = false;

    box.querySelectorAll('[data-fb]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const fb = btn.dataset.fb;
        const body = fb === 'correct'
          ? { id: staging.id, is_correct: true }
          : { id: staging.id, is_correct: false, suggested_label: fb };
        try {
          const res = await fetch('/api/staging/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          const fbBox = box.querySelector('.staging-feedback');
          fbBox.innerHTML = res.ok
            ? '<span class="fb-thanks">Thanks for your feedback!</span>'
            : '<span class="fb-error">Feedback could not be saved (item may already be reviewed).</span>';
        } catch (e) {
          console.error(e);
        }
      });
    });
  };

  // -------------------------------------------------------------------
  // 2. AUTENTIKASI ADMIN
  // -------------------------------------------------------------------
  async function api(url, options = {}) {
    const headers = { ...(options.headers || {}), 'X-Admin-Token': adminToken || '' };
    if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
    const res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
      exitAdminMode(true);
      throw new Error('Admin session expired. Please log in again.');
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    return data;
  }

  function openLoginModal() {
    $('admin-login-error').textContent = '';
    $('admin-password').value = '';
    $('admin-login-modal').hidden = false;
    setTimeout(() => $('admin-password').focus(), 50);
  }

  function closeLoginModal() {
    $('admin-login-modal').hidden = true;
  }

  async function submitLogin(e) {
    e.preventDefault();
    const errEl = $('admin-login-error');
    errEl.textContent = '';
    try {
      const res = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: $('admin-password').value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      adminToken = data.token;
      sessionStorage.setItem(TOKEN_KEY, adminToken);
      closeLoginModal();
      enterAdminMode();
    } catch (err) {
      errEl.textContent = err.message;
    }
  }

  function enterAdminMode() {
    // Matikan kamera kalau sedang menyala (tombol stop ada di app.js)
    const stopBtn = $('btn-stop-webcam');
    if (stopBtn && !stopBtn.disabled) stopBtn.click();

    document.body.classList.add('admin-mode');
    $('btn-admin-login').hidden = true;
    $('admin-active-group').hidden = false;
    window.scrollTo({ top: 0, behavior: 'smooth' });
    refreshDashboard();
    // Kalau re-training sedang berjalan (mis. setelah reload halaman), lanjutkan memantau
    pollRetrainStatus(true);
  }

  async function exitAdminMode(expired = false) {
    if (!expired && adminToken) {
      fetch('/api/admin/logout', { method: 'POST', headers: { 'X-Admin-Token': adminToken } }).catch(() => {});
    }
    adminToken = null;
    sessionStorage.removeItem(TOKEN_KEY);
    stopPolling();
    document.body.classList.remove('admin-mode');
    $('btn-admin-login').hidden = false;
    $('admin-active-group').hidden = true;
    if (expired) alert('Admin session expired. Please log in again.');
  }

  // -------------------------------------------------------------------
  // 3. DASHBOARD: STATISTIK, MODEL & RIWAYAT
  // -------------------------------------------------------------------
  async function refreshDashboard() {
    try {
      const [stats, staging] = await Promise.all([api('/api/admin/stats'), api('/api/admin/staging')]);
      renderStats(stats);
      renderModelInfo(stats.model, stats.pending_retrain);
      renderHistory(stats.history);
      stagingItems = staging.items || [];
      renderStaging();
    } catch (err) {
      console.error(err);
    }
  }

  function renderStats(s) {
    const cards = [
      { label: 'Dataset: Healthy', value: s.raw.sehat, sub: `${s.crowd_approved.sehat} from users`, cls: 'healthy' },
      { label: 'Dataset: Moldy', value: s.raw.terkontaminasi, sub: `${s.crowd_approved.terkontaminasi} from users`, cls: 'moldy' },
      { label: 'Staging Queue', value: s.staging_total, sub: `${s.staging.sehat} healthy · ${s.staging.terkontaminasi} moldy · ${s.staging.campuran} mixed`, cls: 'queue' },
      { label: 'Awaiting Re-train', value: s.pending_retrain, sub: 'approved photos not yet trained', cls: 'pending' },
    ];
    $('admin-stats-grid').innerHTML = cards
      .map((c) => `
        <div class="admin-stat ${c.cls}">
          <span class="admin-stat-label">${c.label}</span>
          <span class="admin-stat-value">${c.value}</span>
          <span class="admin-stat-sub">${escapeHtml(c.sub)}</span>
        </div>`)
      .join('');
  }

  function renderModelInfo(model, pending) {
    const box = $('model-info-box');
    if (!model) {
      box.innerHTML = `
        <div class="model-row"><span>Active model</span><b>v1 (initial training)</b></div>
        <div class="model-row"><span>New photos pending</span><b>${pending}</b></div>
        <p class="hint">No re-training has been run from the dashboard yet.</p>`;
      return;
    }
    box.innerHTML = `
      <div class="model-row"><span>Active model</span><b>v${model.version}</b></div>
      <div class="model-row"><span>Trained at</span><b>${escapeHtml(model.trained_at.replace('T', ' '))}</b></div>
      <div class="model-row"><span>Test accuracy</span><b>${model.accuracy}%</b></div>
      <div class="model-row"><span>Training samples</span><b>${model.n_samples} kernels / ${model.n_images} photos</b></div>
      <div class="model-row"><span>New photos pending</span><b>${pending}</b></div>`;
  }

  function renderHistory(history) {
    const ul = $('admin-history');
    if (!history || history.length === 0) {
      ul.innerHTML = '<li class="hint">No activity yet.</li>';
      return;
    }
    ul.innerHTML = history
      .map((h) => {
        const t = escapeHtml((h.timestamp || '').replace('T', ' '));
        let text = '';
        if (h.action === 'approve') {
          text = h.relabeled
            ? `Approved as <b>${LABELS[h.label]}</b> (relabeled from ${LABELS[h.from_category]})`
            : `Approved as <b>${LABELS[h.label]}</b>`;
        } else if (h.action === 'reject') {
          text = `Rejected a ${LABELS[h.from_category] || ''} photo`;
        } else if (h.action === 'retrain') {
          text = `Re-trained model <b>v${h.version}</b> — accuracy ${h.accuracy}% (+${h.n_new_images} new photos)`;
        } else if (h.action === 'rollback') {
          text = `Rolled back to backup <b>${escapeHtml(h.restored_backup)}</b>`;
        }
        return `<li><span class="hist-time">${t}</span><span>${text}</span></li>`;
      })
      .join('');
  }

  // -------------------------------------------------------------------
  // 4. GALERI KURASI STAGING
  // -------------------------------------------------------------------
  function imgUrl(item, kind) {
    return `/api/admin/staging/image/${item.category}/${encodeURIComponent(item.id)}?kind=${kind}&token=${encodeURIComponent(adminToken)}`;
  }

  function defaultLabel(item) {
    // Prioritas: koreksi user -> kategori staging -> mayoritas biji (untuk campuran)
    if (item.user_feedback && item.user_feedback.suggested_label) return item.user_feedback.suggested_label;
    if (item.category !== 'campuran') return item.category;
    return item.summary.terkontaminasi >= item.summary.sehat ? 'terkontaminasi' : 'sehat';
  }

  function feedbackBadge(item) {
    const fb = item.user_feedback;
    if (!fb) return '';
    if (fb.is_correct) return '<span class="fb-badge ok">User: correct</span>';
    return `<span class="fb-badge warn">User: should be ${LABELS[fb.suggested_label] || '?'}</span>`;
  }

  function renderStaging() {
    // Update jumlah di tab filter
    const counts = { all: stagingItems.length, sehat: 0, terkontaminasi: 0, campuran: 0 };
    stagingItems.forEach((it) => counts[it.category]++);
    document.querySelectorAll('#admin-filter-tabs [data-count]').forEach((el) => {
      el.textContent = counts[el.dataset.count];
    });

    const grid = $('staging-grid');
    const items = activeFilter === 'all' ? stagingItems : stagingItems.filter((it) => it.category === activeFilter);
    if (items.length === 0) {
      grid.innerHTML = '<p class="staging-empty">No photos in this queue. New user detections will appear here automatically.</p>';
      return;
    }

    grid.innerHTML = items
      .map((it) => {
        const s = it.summary;
        const label = defaultLabel(it);
        const conf = it.avg_confidence != null ? `${it.avg_confidence}%` : '-';
        return `
        <article class="staging-card cat-${it.category}" data-id="${escapeHtml(it.id)}">
          <div class="staging-imgs">
            <img src="${imgUrl(it, 'original')}" alt="Original photo" loading="lazy">
            <img src="${imgUrl(it, 'annotated')}" alt="Detection result" loading="lazy">
          </div>
          <div class="staging-body">
            <div class="staging-tags">
              <span class="cat-badge cat-${it.category}">${LABELS[it.category]}</span>
              <span class="src-badge">${SOURCE_LABELS[it.source] || escapeHtml(it.source)}</span>
              ${feedbackBadge(it)}
            </div>
            <p class="staging-meta">${s.sehat} healthy · ${s.terkontaminasi} moldy${s.bukan_jagung ? ` · ${s.bukan_jagung} non-corn` : ''} · avg conf ${conf}</p>
            <p class="staging-time">${escapeHtml((it.created_at || '').replace('T', ' '))}</p>
            <label class="staging-label-select">
              <span>Final label</span>
              <select data-role="label">
                <option value="sehat" ${label === 'sehat' ? 'selected' : ''}>Healthy (sehat)</option>
                <option value="terkontaminasi" ${label === 'terkontaminasi' ? 'selected' : ''}>Mold Contaminated (terkontaminasi)</option>
              </select>
            </label>
            <div class="staging-actions">
              <button type="button" class="btn btn-primary-teal" data-action="approve">Approve to Dataset</button>
              <button type="button" class="btn btn-danger-outline" data-action="reject">Delete</button>
            </div>
          </div>
        </article>`;
      })
      .join('');
  }

  async function handleStagingClick(e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const card = btn.closest('.staging-card');
    const id = card.dataset.id;
    card.querySelectorAll('button').forEach((b) => (b.disabled = true));

    try {
      if (btn.dataset.action === 'approve') {
        const label = card.querySelector('[data-role="label"]').value;
        await api('/api/admin/staging/approve', { method: 'POST', body: JSON.stringify({ id, label }) });
      } else {
        if (!confirm('Delete this photo from the staging queue?')) {
          card.querySelectorAll('button').forEach((b) => (b.disabled = false));
          return;
        }
        await api('/api/admin/staging/reject', { method: 'POST', body: JSON.stringify({ id }) });
      }
      card.classList.add('removing');
      setTimeout(refreshDashboard, 250);
    } catch (err) {
      alert(err.message);
      card.querySelectorAll('button').forEach((b) => (b.disabled = false));
    }
  }

  // -------------------------------------------------------------------
  // 5. RE-TRAINING & ROLLBACK
  // -------------------------------------------------------------------
  function setRetrainButtons(running) {
    $('btn-retrain').disabled = running;
    $('btn-rollback').disabled = running;
    $('retrain-progress').hidden = !running;
  }

  function stopPolling() {
    if (retrainPollId) {
      clearInterval(retrainPollId);
      retrainPollId = null;
    }
  }

  async function startRetrain() {
    if (!confirm('Start re-training the SVM model with the current dataset?')) return;
    $('retrain-result').innerHTML = '';
    try {
      await api('/api/admin/retrain', { method: 'POST' });
      pollRetrainStatus();
    } catch (err) {
      alert(err.message);
    }
  }

  function pollRetrainStatus(silentIfIdle = false) {
    // silentIfIdle: dipakai saat masuk dashboard -- kalau re-training TIDAK sedang
    // berjalan, jangan tampilkan ulang hasil lama. Tapi kalau sempat terlihat
    // berjalan, hasil akhirnya tetap ditampilkan.
    stopPolling();
    let sawRunning = false;
    const tick = async () => {
      try {
        const st = await api('/api/admin/retrain/status');
        if (st.running) {
          sawRunning = true;
          setRetrainButtons(true);
          $('retrain-progress-fill').style.width = `${st.progress}%`;
          $('retrain-progress-text').textContent = `${st.progress}% — ${st.message}`;
          return;
        }
        stopPolling();
        setRetrainButtons(false);
        const show = !silentIfIdle || sawRunning;
        if (st.stage === 'done' && st.result && show) {
          renderRetrainResult(st.result);
          refreshDashboard();
        } else if (st.stage === 'error' && show) {
          $('retrain-result').innerHTML = `<div class="retrain-error">Re-training failed: ${escapeHtml(st.error)}</div>`;
        }
      } catch (err) {
        stopPolling();
        setRetrainButtons(false);
      }
    };
    tick();
    retrainPollId = setInterval(tick, 1500);
  }

  function renderRetrainResult(r) {
    const prev = r.previous_accuracy_same_test;
    let diffHtml = '';
    if (prev != null) {
      const diff = +(r.accuracy - prev).toFixed(2);
      const cls = diff > 0 ? 'up' : diff < 0 ? 'down' : 'same';
      const sign = diff > 0 ? '+' : '';
      diffHtml = `<span class="acc-diff ${cls}">${sign}${diff}%</span>`;
    }
    const f1 = r.f1_per_class || {};
    $('retrain-result').innerHTML = `
      <div class="retrain-compare">
        <div class="acc-box old">
          <span>Previous model</span>
          <b>${prev != null ? prev + '%' : '-'}</b>
        </div>
        <div class="acc-arrow">&#8594;</div>
        <div class="acc-box new">
          <span>New model v${r.version}</span>
          <b>${r.accuracy}%</b>
        </div>
        ${diffHtml}
      </div>
      <p class="hint">Both models evaluated on the same held-out test set (20%).
        New photos used: <b>${r.n_new_images}</b>${r.n_skipped_new_images ? ` (${r.n_skipped_new_images} skipped: no kernel segmented)` : ''} ·
        F1 healthy ${f1.sehat ?? '-'}% · F1 moldy ${f1.terkontaminasi ?? '-'}% ·
        C=${escapeHtml(r.best_params.C)}, gamma=${escapeHtml(r.best_params.gamma)} · ${r.duration_sec}s</p>
      ${prev != null && r.accuracy < prev ? '<p class="retrain-warn">Accuracy dropped. Consider reviewing recently approved photos or use "Rollback to Previous Model".</p>' : ''}`;
  }

  async function rollbackModel() {
    if (!confirm('Restore the previous model version? The current model will be discarded.')) return;
    try {
      const r = await api('/api/admin/rollback', { method: 'POST' });
      $('retrain-result').innerHTML = `<div class="retrain-ok">Restored model backup <b>${escapeHtml(r.restored)}</b>.</div>`;
      refreshDashboard();
    } catch (err) {
      alert(err.message);
    }
  }

  // -------------------------------------------------------------------
  // 6. EVENT LISTENERS & INIT
  // -------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', async () => {
    $('btn-admin-login').addEventListener('click', openLoginModal);
    $('btn-admin-cancel').addEventListener('click', closeLoginModal);
    $('admin-login-modal').addEventListener('click', (e) => {
      if (e.target.id === 'admin-login-modal') closeLoginModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !$('admin-login-modal').hidden) closeLoginModal();
    });
    $('admin-login-form').addEventListener('submit', submitLogin);
    $('btn-admin-logout').addEventListener('click', () => exitAdminMode(false));
    $('btn-admin-refresh').addEventListener('click', refreshDashboard);
    $('staging-grid').addEventListener('click', handleStagingClick);
    $('btn-retrain').addEventListener('click', startRetrain);
    $('btn-rollback').addEventListener('click', rollbackModel);

    document.querySelectorAll('#admin-filter-tabs .filter-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('#admin-filter-tabs .filter-tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        activeFilter = tab.dataset.filter;
        renderStaging();
      });
    });

    // Pulihkan sesi admin setelah reload halaman (token masih berlaku?)
    if (adminToken) {
      try {
        const res = await fetch('/api/admin/check', { headers: { 'X-Admin-Token': adminToken } });
        const data = await res.json();
        if (data.valid) enterAdminMode();
        else resetAdminState();
      } catch (e) {
        resetAdminState();
      }
    }
  });

  function resetAdminState() {
    adminToken = null;
    sessionStorage.removeItem(TOKEN_KEY);
  }
})();
