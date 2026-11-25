/* TomatoGuard 全站共享脚本：导航 / 页脚 / 图标 / Toast / 灯箱 / 图表主题 / 数据加载 */

(function () {
  "use strict";

  const TG = (window.TG = window.TG || {});

  /* ---------------------------------------------------------------- 图标 */

  const ICONS = {
    leaf: '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" y1="12" x2="17" y2="12"/>',
    book: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    chart: '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6" rx="1"/><rect x="12" y="8" width="3" height="10" rx="1"/><rect x="17" y="5" width="3" height="13" rx="1"/>',
    shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    chevron: '<polyline points="6 9 12 15 18 9"/>',
    arrow: '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    refresh: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    check: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    alert: '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    clock: '<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/>',
    cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    menu: '<line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>',
    expand: '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>',
    copy: '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    info: '<circle cx="12" cy="12" r="9"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    server: '<rect x="2" y="3" width="20" height="8" rx="2"/><rect x="2" y="13" width="20" height="8" rx="2"/><line x1="6" y1="7" x2="6.01" y2="7"/><line x1="6" y1="17" x2="6.01" y2="17"/>',
    git: '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="8" r="2"/><path d="M6 8v8"/><path d="M18 10c0 3-3 4-6 4H8"/>',
    spark: '<path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"/>',
    activity: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    flask: '<path d="M9 3h6"/><path d="M10 3v6L4.5 18.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3"/><line x1="7.5" y1="14" x2="16.5" y2="14"/>',
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
    folder: '<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7l-2-2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2z"/>',
    filter: '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    plus: '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
    user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    key: '<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>',
  };

  TG.icon = function (name, size) {
    const path = ICONS[name] || ICONS.info;
    return (
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ` +
      `stroke-linecap="round" stroke-linejoin="round"${size ? ` width="${size}" height="${size}"` : ""}>` +
      path +
      "</svg>"
    );
  };

  /* ---------------------------------------------------------------- 导航 */

  const NAV = [
    { key: "home", label: "首页", href: "index.html" },
    {
      key: "detection",
      label: "智能识别",
      href: "detect.html",
      children: [
        { label: "单图识别", href: "detect.html" },
        { label: "批量识别", href: "batch.html" },
        { label: "识别历史", href: "history.html" },
      ],
    },
    { key: "library", label: "病害百科", href: "library.html" },
    {
      key: "analysis",
      label: "模型分析",
      href: "models.html",
      children: [
        { label: "模型概览", href: "models.html" },
        { label: "模型性能对比", href: "compare.html" },
        { label: "鲁棒性实验", href: "robustness.html" },
        { label: "跨域实验", href: "crossdomain.html" },
        { label: "Grad-CAM 分析", href: "gradcam.html" },
      ],
    },
    {
      key: "data",
      label: "数据中心",
      href: "dataset.html",
      children: [
        { label: "数据集介绍", href: "dataset.html" },
        { label: "实验报告", href: "reports.html" },
      ],
    },
    { key: "about", label: "关于系统", href: "about.html" },
  ];

  const PAGE_KEY = {
    "index.html": "home",
    "detect.html": "detection",
    "batch.html": "detection",
    "history.html": "detection",
    "library.html": "library",
    "disease.html": "library",
    "models.html": "analysis",
    "compare.html": "analysis",
    "robustness.html": "analysis",
    "crossdomain.html": "analysis",
    "gradcam.html": "analysis",
    "dataset.html": "data",
    "reports.html": "data",
    "about.html": "about",
  };

  function currentPage() {
    const name = location.pathname.split("/").pop() || "index.html";
    return PAGE_KEY[name] || "home";
  }

  TG.renderNav = function (user) {
    const mount = document.getElementById("siteNav");
    if (!mount) return;
    const active = currentPage();

    const items = NAV.map((item) => {
      const isActive = item.key === active;
      if (!item.children) {
        return `<div class="nav-item"><a class="nav-link${isActive ? " active" : ""}" href="${item.href}">${item.label}</a></div>`;
      }
      const sub = item.children
        .map(
          (c) =>
            `<a href="${c.href}" class="${location.pathname.endsWith(c.href) ? "active" : ""}">${c.label}</a>`
        )
        .join("");
      return (
        `<div class="nav-item has-menu">` +
        `<a class="nav-link${isActive ? " active" : ""}" href="${item.href}">${item.label}` +
        `<svg class="caret" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>` +
        `</a><div class="nav-menu">${sub}</div></div>`
      );
    }).join("");

    /* 右上角：状态胶囊 + 用户菜单 */
    let userHtml;
    if (user) {
      const initial = (user.username || "U").charAt(0).toUpperCase();
      const roleLabel = user.role === "admin" ? "管理员" : "普通用户";
      userHtml = (
        `<div class="user-menu" id="userMenu" data-open="false">` +
        `<button class="user-btn" id="userBtn" aria-haspopup="menu" aria-label="用户菜单">` +
        `<span class="avatar" title="${TG.escape(user.username)}">${initial}</span>` +
        `<span class="user-name">${TG.escape(user.username)}</span>` +
        `<span class="caret">${TG.icon("chevron", 14)}</span>` +
        `</button>` +
        `<div class="user-dropdown" role="menu">` +
        `<div class="user-dropdown-head">` +
        `<div><b>${TG.escape(user.username)}</b></div>` +
        `<div class="muted" style="font-size:12px">${roleLabel}</div>` +
        `</div>` +
        `<div class="user-dropdown-body">` +
        `<a class="user-dropdown-item" href="about.html"><span>${TG.icon("info", 15)}</span>个人中心（关于系统）</a>` +
        `<a class="user-dropdown-item" href="#" id="logoutBtn"><span>${TG.icon("logout", 15)}</span>退出登录</a>` +
        `</div>` +
        `</div></div>`
      );
    } else {
      userHtml = (
        `<div class="user-menu user-menu--guest">` +
        `<a class="btn btn-ghost btn-sm" href="login.html">登录</a>` +
        `<a class="btn btn-primary btn-sm" href="register.html">注册</a>` +
        `</div>`
      );
    }

    mount.className = "site-nav";
    mount.innerHTML =
      `<div class="nav-inner">` +
      `<a class="brand" href="index.html" aria-label="TomatoGuard 首页">` +
      `<span class="brand-mark">${TG.icon("leaf")}</span>` +
      `<span class="brand-text"><b>TomatoGuard</b><small>番茄叶片病害智能识别平台</small></span></a>` +
      `<nav class="nav-links" id="navLinks">${items}</nav>` +
      `<div class="nav-right">` +
      `<span class="status-pill" id="sysStatus"><i></i>系统在线</span>` +
      userHtml +
      `<button class="nav-toggle" id="navToggle" aria-label="打开菜单" aria-expanded="false">${TG.icon("menu", 20)}</button>` +
      `</div></div>`;

    const toggle = document.getElementById("navToggle");
    const links = document.getElementById("navLinks");
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    mount.querySelectorAll(".has-menu > .nav-link").forEach((link) => {
      link.addEventListener("click", (event) => {
        if (window.innerWidth > 991) return;
        event.preventDefault();
        link.parentElement.classList.toggle("open");
      });
    });

    links.querySelectorAll("a").forEach((a) => {
      a.addEventListener("click", () => {
        if (window.innerWidth <= 991) {
          links.classList.remove("open");
          toggle.setAttribute("aria-expanded", "false");
        }
      });
    });

    /* 用户下拉交互 */
    const userMenu = document.getElementById("userMenu");
    if (userMenu && user) {
      const btn = document.getElementById("userBtn");
      const logout = document.getElementById("logoutBtn");
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = userMenu.getAttribute("data-open") === "true";
        userMenu.setAttribute("data-open", String(!open));
      });
      document.addEventListener("click", (e) => {
        if (!userMenu.contains(e.target)) userMenu.setAttribute("data-open", "false");
      });
      if (logout) {
        logout.addEventListener("click", (e) => {
          e.preventDefault();
          TG.auth.logout();
        });
      }
    }

    checkHealth();
  };

  function checkHealth() {
    const pill = document.getElementById("sysStatus");
    if (!pill) return;
    fetch("api/health", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("offline"))))
      .then((data) => {
        pill.classList.remove("offline");
        pill.innerHTML = `<i></i>${data.model} · 在线`;
        pill.title = `${data.device} · ${data.params_m}M 参数`;
      })
      .catch(() => {
        pill.classList.add("offline");
        pill.innerHTML = `<i></i>推理服务未启动`;
        pill.title = "请运行 python web/server.py";
      });
  }

  /* ---------------------------------------------------------------- 页脚 */

  TG.renderFooter = function () {
    const mount = document.getElementById("siteFooter");
    if (!mount) return;
    mount.className = "site-footer";
    mount.innerHTML =
      `<div class="container footer-grid">` +
      `<div class="footer-brand">` +
      `<a class="brand" href="index.html"><span class="brand-mark">${TG.icon("leaf")}</span>` +
      `<span class="brand-text"><b>TomatoGuard</b><small>番茄叶片病害智能识别平台</small></span></a>` +
      `<p>基于迁移学习的番茄叶片病害识别与鲁棒性分析平台，覆盖智能识别、病害知识、模型实验与可解释性分析。</p>` +
      `</div>` +
      `<div class="footer-col"><h4>产品功能</h4>` +
      `<a href="detect.html">单图识别</a><a href="batch.html">批量识别</a>` +
      `<a href="history.html">识别历史</a><a href="library.html">病害百科</a></div>` +
      `<div class="footer-col"><h4>模型研究</h4>` +
      `<a href="models.html">模型概览</a><a href="compare.html">性能对比</a>` +
      `<a href="robustness.html">鲁棒性实验</a><a href="crossdomain.html">跨域实验</a>` +
      `<a href="gradcam.html">Grad-CAM</a></div>` +
      `<div class="footer-col"><h4>数据与项目</h4>` +
      `<a href="dataset.html">数据集中心</a><a href="reports.html">实验报告</a>` +
      `<a href="about.html">关于系统</a>` +
      `<span>数据来源：PlantVillage</span></div>` +
      `</div>` +
      `<div class="container footer-bottom">` +
      `<span>© 2026 TomatoGuard · 植物叶片病害识别课程项目</span>` +
      `<span class="mono">ResNet-18 · 6 Classes · PlantVillage</span>` +
      `</div>`;
  };

  /* ---------------------------------------------------------------- Toast */

  TG.toast = function (message, type) {
    let stack = document.querySelector(".toast-stack");
    if (!stack) {
      stack = document.createElement("div");
      stack.className = "toast-stack";
      document.body.appendChild(stack);
    }
    const el = document.createElement("div");
    el.className = "toast " + (type || "");
    el.textContent = message;
    stack.appendChild(el);
    setTimeout(() => {
      el.classList.add("hide");
      setTimeout(() => el.remove(), 220);
    }, 3200);
  };

  /* ---------------------------------------------------------------- 灯箱 */

  TG.openLightbox = function (src, alt) {
    let box = document.getElementById("lightbox");
    if (!box) {
      box = document.createElement("div");
      box.id = "lightbox";
      box.className = "lightbox";
      box.innerHTML = `<button class="close" aria-label="关闭">×</button><img alt="">`;
      document.body.appendChild(box);
      const close = () => box.classList.remove("open");
      box.querySelector(".close").addEventListener("click", close);
      box.addEventListener("click", (e) => {
        if (e.target === box) close();
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") close();
      });
    }
    const img = box.querySelector("img");
    img.src = src;
    img.alt = alt || "图片预览";
    box.classList.add("open");
  };

  /* ---------------------------------------------------------------- 格式化 */

  TG.pct = function (value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return (Number(value) * 100).toFixed(digits === undefined ? 2 : digits) + "%";
  };

  TG.num = function (value) {
    if (value === null || value === undefined) return "—";
    return Number(value).toLocaleString("zh-CN");
  };

  TG.escape = function (text) {
    return String(text === null || text === undefined ? "" : text).replace(/[&<>"']/g, (s) => {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[s];
    });
  };

  TG.time = function (iso) {
    const d = iso ? new Date(iso) : new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };

  TG.bytes = function (bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(2) + " MB";
  };

  /* ---------------------------------------------------------------- 数据 */

  let siteDataPromise = null;
  TG.loadSiteData = function () {
    if (!siteDataPromise) {
      siteDataPromise = fetch("api/site-data", { cache: "no-store" })
        .then((r) => {
          if (!r.ok) throw new Error("site-data " + r.status);
          return r.json();
        })
        .catch(() => fetch("data/site_data.json").then((r) => r.json()));
    }
    return siteDataPromise;
  };

  /* ---------------------------------------------------------------- 图表 */

  const PALETTE = ["#237a57", "#67a879", "#f28b35", "#8fb8a0", "#c96a12", "#4a564f"];

  TG.chart = function (el, option) {
    if (!window.echarts || !el) return null;
    const chart = window.echarts.init(el, null, { renderer: "canvas" });
    const base = {
      color: PALETTE,
      textStyle: { fontFamily: getComputedStyle(document.body).fontFamily, color: "#4a564f" },
      grid: { left: 48, right: 24, top: 42, bottom: 40, containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(32,43,37,.94)",
        borderWidth: 0,
        textStyle: { color: "#fff", fontSize: 12 },
        axisPointer: { type: "line", lineStyle: { color: "#c9d3cd" } },
      },
      legend: { top: 4, icon: "roundRect", itemWidth: 10, itemHeight: 10, textStyle: { color: "#7b8580", fontSize: 12 } },
    };
    const merged = Object.assign({}, base, option);
    merged.grid = Object.assign({}, base.grid, option.grid || {});
    chart.setOption(merged);
    window.addEventListener("resize", () => chart.resize());
    return chart;
  };

  TG.palette = PALETTE;

  TG.download = function (href, filename) {
    const a = document.createElement("a");
    a.href = href;
    a.download = filename || "download";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  /* ---------------------------------------------------------------- Auth */

  const CURRENT_USER_KEY = "tg_current_user";

  TG.auth = {
    me: function () {
      return fetch("api/auth/me", { credentials: "same-origin", cache: "no-store" })
        .then((r) => r.ok ? r.json() : { ok: false, user: null })
        .catch(() => ({ ok: false, user: null }));
    },
    login: function (username, password) {
      return fetch("api/auth/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password: password }),
      }).then((r) => {
        if (r.status === 401 || r.status === 400) {
          return r.json().then((d) => { throw new Error(d.detail || "登录失败"); });
        }
        if (!r.ok) throw new Error("登录失败");
        return r.json();
      });
    },
    register: function (username, password) {
      return fetch("api/auth/register", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password: password }),
      }).then((r) => {
        if (!r.ok) return r.json().then((d) => { throw new Error(d.detail || "注册失败"); });
        return r.json();
      });
    },
    logout: function () {
      fetch("api/auth/logout", { method: "POST", credentials: "same-origin", cache: "no-store" }).catch(() => {});
      localStorage.removeItem(CURRENT_USER_KEY);
      TG.currentUser = null;
      TG.renderNav(null);
      TG.applyRole(null);
      if (!location.pathname.endsWith("/login.html") && !location.pathname.endsWith("/register.html")) {
        TG.toast("已退出登录", "ok");
      }
    },
  };

  /* 根据角色显示 / 隐藏页面上 data-role="admin" 等元素 */
  TG.applyRole = function (user) {
    document.querySelectorAll("[data-role]").forEach((el) => {
      const need = (el.getAttribute("data-role") || "").toLowerCase();
      if (!need) return;
      const ok = user && (need === "any" || need === "user" || (need === "admin" && user.role === "admin"));
      el.style.display = ok ? "" : "none";
      el.hidden = !ok;
    });
  };

  /* ---------------------------------------------------------------- 初始化 */

  function init() {
    document.querySelectorAll("[data-icon]").forEach((el) => {
      el.innerHTML = TG.icon(el.getAttribute("data-icon"), el.getAttribute("data-size"));
    });

    /* 先 renderNav（空壳，不依赖用户），再异步 me() 驱动用户态与 applyRole */
    TG.renderNav(null);
    TG.renderFooter();

    /* 本地缓存当前用户，避免首次加载 me() 失败时 nav 完全空白 */
    try {
      const cached = JSON.parse(localStorage.getItem(CURRENT_USER_KEY) || "null");
      if (cached && cached.username) {
        TG.renderNav(cached);
        TG.applyRole(cached); /* 先用本地角色渲染，避免受限内容闪一下 */
      }
    } catch (_) {}

    TG.auth.me().then((data) => {
      const user = data && data.user ? data.user : null;
      TG.currentUser = user;
      if (user) localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
      else localStorage.removeItem(CURRENT_USER_KEY);
      TG.renderNav(user);
      TG.applyRole(user);
    });

    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add("in");
              io.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.08 }
      );
      document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
    } else {
      document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
