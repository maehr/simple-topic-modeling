import { Er as e, Ii as t, ds as n, fs as r, hs as i, ps as a } from "./index-g7ewbxvi-0eszC5oh.js";
//#region ../../node_modules/react-dom/client.js
var o = /* @__PURE__ */ a(((e) => {
	var t = n();
	e.createRoot = t.createRoot, e.hydrateRoot = t.hydrateRoot;
})), s = /* @__PURE__ */ i(r(), 1), c = o(), l = {
	stliteLib: new URL("./wheels/stlite_lib-0.1.0-py3-none-any.whl", import.meta.url).href,
	streamlit: new URL("./wheels/streamlit-1.62.0-cp313-none-any.whl", import.meta.url).href
};
//#endregion
//#region src/options.ts
function u(e) {
	if (e == null) return {};
	let t = {};
	return Object.keys(e).forEach((n) => {
		let r = e[n];
		if (typeof r == "string" || ArrayBuffer.isView(r)) {
			t[n] = { data: r };
			return;
		}
		if (typeof r == "object") {
			if ("data" in r) {
				t[n] = r;
				return;
			}
			if ("url" in r) {
				t[n] = {
					...r,
					url: d(r.url)
				};
				return;
			}
		}
		throw Error(`Invalid file value: ${r}`);
	}), t;
}
function d(e) {
	return new URL(e, window.location.href).href;
}
function f(e) {
	return e == null ? [] : e.map((e) => "buffer" in e ? e : {
		...e,
		url: d(e.url)
	});
}
var p = "streamlit_app.py";
function m(e) {
	if (typeof e == "string") {
		let t = e;
		return {
			kernelOptions: {
				entrypoint: p,
				files: { [p]: { data: t } },
				archives: [],
				requirements: [],
				prebuiltPackageNames: []
			},
			toastOptions: {
				disableProgressToasts: !1,
				disableErrorToasts: !1,
				disableModuleAutoLoadToasts: !1
			},
			styleOptions: {
				disableDocumentStyles: !1,
				styleNonce: void 0
			}
		};
	}
	let t = u(e.files), n = f(e.archives), r = e.entrypoint;
	if (r == null) throw Error("The `entrypoint` field is required.");
	return {
		kernelOptions: {
			entrypoint: r,
			files: t,
			archives: n,
			requirements: e.requirements || [],
			prebuiltPackageNames: e.prebuiltPackageNames || [],
			hostConfigProperties: e.hostConfig,
			pyodideUrl: e.pyodideUrl,
			streamlitConfig: e.streamlitConfig,
			wheelUrls: e.wheelUrls,
			idbfsMountpoints: e.idbfsMountpoints,
			sharedWorker: e.sharedWorker,
			env: e.env,
			languageServer: e.languageServer
		},
		toastOptions: {
			disableProgressToasts: e.disableProgressToasts || !1,
			disableErrorToasts: e.disableErrorToasts || !1,
			disableModuleAutoLoadToasts: e.disableModuleAutoLoadToasts || !1
		},
		styleOptions: {
			disableDocumentStyles: e.disableDocumentStyles ?? !1,
			styleNonce: e.styleNonce
		}
	};
}
//#endregion
//#region ../../node_modules/react/cjs/react-jsx-runtime.production.min.js
var h = /* @__PURE__ */ a(((e) => {
	var t = r(), n = Symbol.for("react.element"), i = Object.prototype.hasOwnProperty, a = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, o = {
		key: !0,
		ref: !0,
		__self: !0,
		__source: !0
	};
	function s(e, t, r) {
		var s, c = {}, l = null, u = null;
		for (s in r !== void 0 && (l = "" + r), t.key !== void 0 && (l = "" + t.key), t.ref !== void 0 && (u = t.ref), t) i.call(t, s) && !o.hasOwnProperty(s) && (c[s] = t[s]);
		if (e && e.defaultProps) for (s in t = e.defaultProps, t) c[s] === void 0 && (c[s] = t[s]);
		return {
			$$typeof: n,
			type: e,
			key: l,
			ref: u,
			props: c,
			_owner: a.current
		};
	}
	e.jsx = s;
})), g = (/* @__PURE__ */ a(((e, t) => {
	t.exports = h();
})))(), _ = "classic";
function v(n, r = document.body) {
	let { kernelOptions: i, toastOptions: a, styleOptions: o } = m(n), u = t({
		...i,
		wheelUrls: i.wheelUrls ?? l,
		workerType: _
	}), d = (0, c.createRoot)(r);
	return d.render(/* @__PURE__ */ (0, g.jsx)(s.StrictMode, { children: /* @__PURE__ */ (0, g.jsx)(e, {
		kernel: u,
		...a,
		...o
	}) })), {
		unmount: () => {
			u.dispose(), d.unmount();
		},
		install: (e, t) => u.install(e, t),
		addMockPackage: (e, t, n, r) => u.addMockPackage(e, t, n, r),
		writeFile: (e, t, n) => u.writeFile(e, t, n),
		renameFile: (e, t) => u.renameFile(e, t),
		unlink: (e) => u.unlink(e),
		readFile: (e, t) => u.readFile(e, t),
		getCodeCompletion: (e, t) => u.getCodeCompletion(e, t),
		runPython: (e) => u.runPython(e),
		_kernel: u
	};
}
//#endregion
//#region ../common/dist/requirements-txt.js
var y = /\s#.*$/;
function b(e) {
	return e.split("\n").filter((e) => !e.startsWith("#")).map((e) => e.replace(y, "")).map((e) => e.trim()).filter((e) => e !== "");
}
//#endregion
//#region ../../node_modules/plain-tag/esm/index.js
function x(e) {
	for (var t = e[0], n = 1, r = arguments.length; n < r; n++) t += arguments[n] + e[n];
	return t;
}
//#endregion
//#region ../../node_modules/codedent/esm/index.js
var S = {
	object(...e) {
		return this.string(x(...e));
	},
	string(e) {
		for (let t of e.split(/[\r\n]+/)) if (t.trim().length) {
			/^(\s+)/.test(t) && (e = e.replace(RegExp("^" + RegExp.$1, "gm"), ""));
			break;
		}
		return e;
	}
}, C = (e, ...t) => S[typeof e](e, ...t);
//#endregion
//#region src/custom-element.ts
function w(e) {
	try {
		let t = new URL(e, window.location.href).pathname.split("/").pop() || "";
		return t.endsWith(".py") ? t : "streamlit_app.py";
	} catch {
		let t = e.split("?")[0].split("/").pop() || "";
		return t.endsWith(".py") ? t : "streamlit_app.py";
	}
}
function T(e) {
	class t extends HTMLElement {
		constructor(...e) {
			super(...e), this._controller = null, this.parseOptions = () => {
				let e = {}, t = "", n = [], r = {}, i = this.getAttribute("src"), a = null, o = !1, s = null;
				i && (a = w(i), s = a, o = !0, e[a] = { url: d(i) });
				let c = [];
				if (this.childNodes.forEach((r) => {
					if (r instanceof Text) {
						if (o) return;
						let e = r.textContent;
						if (e == null || e.replace(/\s*/g, "") === "") return;
						c.push(e);
						return;
					}
					if (r instanceof HTMLElement) switch (r.tagName) {
						case "APP-FILE": {
							let t = r.getAttribute("name");
							if (!t) throw Error("Attribute 'name' is required for <app-file>");
							if (e[t]) throw Error(t === s ? `File with name '${t}' conflicts with the 'src' attribute. The 'src' attribute already defines a file named '${t}'. Remove the conflicting <app-file> or use a different filename.` : `File with name '${t}' already exists`);
							if (!o && r.hasAttribute("entrypoint")) {
								if (a) throw Error("Multiple entrypoints are not allowed");
								a = t;
							}
							let n = r.getAttribute("url"), i = r.getAttribute("encoding");
							n ? e[t] = {
								url: n,
								opts: i ? { encoding: i } : void 0
							} : e[t] = r.textContent ? C(r.textContent) : "";
							return;
						}
						case "APP-REQUIREMENTS":
							t += r.textContent ? C(r.textContent) : "";
							return;
						case "APP-ARCHIVE": {
							let e = r.getAttribute("url"), t = r.getAttribute("format");
							if (!e || !t) throw Error("Attributes 'url' and 'format' are required for <app-archive>");
							n.push({
								url: e,
								format: t
							});
							return;
						}
					}
				}), a === null) {
					if (c.length === 0) throw Error("No content found");
					a = "streamlit_app.py", e[a] = C(c[0]);
				}
				if (!a) throw Error("Entrypoint is required");
				let l = {};
				return this.hasAttribute("shared-worker") && (l.sharedWorker = !0), {
					...l,
					entrypoint: a,
					files: e,
					requirements: b(t),
					archives: n,
					streamlitConfig: r
				};
			};
		}
		connectedCallback() {
			let t = this.parseOptions(), n = document.createElement("div");
			n.classList.add("stlite-app-container"), this.appendChild(n), this._controller = e(t, n), this.style.display = "block";
		}
		disconnectedCallback() {
			this._controller?.unmount();
		}
	}
	customElements.define("streamlit-app", t);
}
//#endregion
//#region src/index.ts
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", () => {
	T(v);
}) : T(v);
//#endregion
export { v as mount };

//# sourceMappingURL=stlite.js.map