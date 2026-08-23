//
//  A WKWebView that loads one page, evaluates one expression, and prints what it resolves
//  to. Nothing else.
//
//  This exists because two of Phase 7's numbers are only true if they are true *in
//  WKWebView*. `overview.md` §5.1 warns that the maximum `gl_PointSize` is
//  driver-dependent and "can be as low as 64 px" — a claim about ANGLE-on-Metal inside
//  WebKit, which Chrome cannot answer and which decides whether the whole layer is
//  `THREE.Points` or `InstancedMesh`. The frame-time exit criterion has the same shape:
//  60 fps in a desktop browser is not evidence about the engine this app ships in.
//
//  Tauri's own webview is this class, configured by `wry`. A measurement taken here is
//  taken on the same renderer, the same GPU process, and the same Metal backend the
//  shipped app will use, without needing the app — which matters because the app cannot
//  currently be driven headlessly and because these two questions must be answerable
//  before the code that depends on their answers is written.
//
//  Build and run through `scripts/webview_eval.mjs`, which compiles this on demand.
//
//      swiftc -O scripts/webview_eval.swift -o build/webview-eval
//      build/webview-eval --url http://localhost:5199/ --expr 'probe()' --timeout 30
//
//  `--expr` may evaluate to a value or to a Promise; both are awaited. The result is
//  printed to stdout as JSON and the process exits 0. Anything else — a JS exception, a
//  failed navigation, a timeout — goes to stderr with a non-zero exit, so a caller can
//  tell "the measurement says X" from "there was no measurement".
//

import AppKit
import WebKit

// MARK: - Arguments

func argument(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("webview-eval: \(message)\n".data(using: .utf8)!)
    exit(1)
}

guard let urlString = argument("--url"), let url = URL(string: urlString) else {
    fail("usage: webview-eval --url <url> --expr <js> [--timeout <seconds>]")
}
guard let expression = argument("--expr") else {
    fail("usage: webview-eval --url <url> --expr <js> [--timeout <seconds>]")
}
let timeout = Double(argument("--timeout") ?? "60") ?? 60

// A window large enough that a 50,000-point cloud is actually rasterizing pixels. A 1×1
// offscreen view would measure vertex processing and nothing else, and fill rate is
// precisely what `antialias: false` and the additive-blend decision in §5.4 are about.
let width = Double(argument("--width") ?? "1280") ?? 1280
let height = Double(argument("--height") ?? "800") ?? 800

// MARK: - Bridge
//
// `evaluateJavaScript` can only hand back JSON-serializable values, and it does not await
// promises. Rather than poll, the page resolves and posts the result to this handler.

final class Bridge: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    private let expression: String
    private var finished = false

    init(expression: String) {
        self.expression = expression
    }

    func webView(_ webView: WKWebView, didFinish _: WKNavigation!) {
        // `await` at the top level of `evaluateJavaScript` is not allowed, so the
        // expression is wrapped in an async IIFE that reports through the handler. A
        // rejection is reported too — a measurement harness that threw and then timed out
        // would be indistinguishable from one that hung.
        let script = """
        (async () => {
          const post = (m) => window.webkit.messageHandlers.result.postMessage(m);
          try {
            post({ ok: true, value: await (\(expression)) });
          } catch (e) {
            post({ ok: false, error: String((e && e.stack) || e) });
          }
        })();
        // The IIFE evaluates to a Promise, and `evaluateJavaScript` cannot marshal one --
        // it would report "unsupported type" and this process would exit before the result
        // ever arrived through the handler. A trailing literal makes the completion value
        // something WebKit can hand back, and it is ignored.
        undefined;
        """
        webView.evaluateJavaScript(script) { _, error in
            if let error = error { self.finish(error: "evaluate: \(error)") }
        }
    }

    func webView(_: WKWebView, didFail _: WKNavigation!, withError error: Error) {
        finish(error: "navigation failed: \(error.localizedDescription)")
    }

    func webView(_: WKWebView, didFailProvisionalNavigation _: WKNavigation!, withError error: Error) {
        finish(error: "navigation failed: \(error.localizedDescription)")
    }

    func userContentController(_: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any] else {
            return finish(error: "handler received \(type(of: message.body)), expected an object")
        }
        // A long measurement that stops making progress is otherwise a bare timeout. Pages
        // may post `{ progress: "..." }` as often as they like; it goes to stderr and the
        // run continues. Only a message carrying `ok` ends the process.
        if let progress = body["progress"] as? String {
            FileHandle.standardError.write("  · \(progress)\n".data(using: .utf8)!)
            return
        }
        if body["ok"] as? Bool != true {
            return finish(error: (body["error"] as? String) ?? "the page reported a failure")
        }
        guard let value = body["value"] else { return finish(error: "the page resolved to nothing") }
        guard JSONSerialization.isValidJSONObject(value) || value is String else {
            return finish(error: "the page resolved to a value that is not JSON")
        }
        if let text = value as? String {
            finish(output: text)
        } else if let data = try? JSONSerialization.data(withJSONObject: value),
                  let text = String(data: data, encoding: .utf8) {
            finish(output: text)
        } else {
            finish(error: "could not serialize the page's result")
        }
    }

    private func finish(output: String) {
        guard !finished else { return }
        finished = true
        print(output)
        exit(0)
    }

    private func finish(error: String) {
        guard !finished else { return }
        finished = true
        FileHandle.standardError.write("webview-eval: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

// MARK: - Run

let app = NSApplication.shared
// `.regular` rather than `.accessory`, which is what this started as. An accessory app
// cannot become frontmost, and a window that is not frontmost is a window WebKit is willing
// to call hidden — see the note on the window below.
app.setActivationPolicy(.regular)

let bridge = Bridge(expression: expression)
let configuration = WKWebViewConfiguration()
configuration.userContentController.add(bridge, name: "result")
configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")

let frame = NSRect(x: 0, y: 0, width: width, height: height)
let webView = WKWebView(frame: frame, configuration: configuration)
webView.navigationDelegate = bridge

let window = NSWindow(contentRect: frame, styleMask: [.titled], backing: .buffered, defer: false)
window.contentView = webView
window.title = "audiobank measurement"
// Frame timing only means anything if this window is genuinely being composited, and
// getting that right cost more attempts than it should have. WebKit suspends
// `requestAnimationFrame` whenever it believes the page is hidden, and it decides that from
// the window's occlusion state -- so a window merely sitting behind a full-screen terminal,
// or on another Space, reports `document.visibilityState === 'hidden'` and runs **zero**
// frames. The harness then posts no progress and resolves nothing, and the failure looks
// exactly like a hang inside the page rather than like a window being ignored.
//
// Join every Space so switching desktops does not hide it, and come to the front.
window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
window.orderFrontRegardless()
window.makeKeyAndOrderFront(nil)
NSApp.activate(ignoringOtherApps: true)

// And the part that actually settles it. Coming to the front is not sufficient on its own:
// whether WebKit calls this window occluded depends on what else is on the desktop, which
// makes the measurement's ability to run depend on the state of the machine. `-[WKWebView
// _setWindowOcclusionDetectionEnabled:]` is SPI, which would be unacceptable in the shipped
// app and is fine here: this file is a developer measurement tool in `scripts/`, it is never
// bundled, and the alternative is a benchmark that silently declines to run. Guarded by
// `responds(to:)` so a future WebKit that drops it degrades to the settings above.
let disableOcclusion = NSSelectorFromString("_setWindowOcclusionDetectionEnabled:")
if webView.responds(to: disableOcclusion) {
    webView.perform(disableOcclusion, with: false as NSNumber)
}

// Belt and braces: if the system reports the window occluded anyway, say so rather than
// silently producing frame times from a throttled compositor.
var occlusionObserver: NSObjectProtocol?
occlusionObserver = NotificationCenter.default.addObserver(
    forName: NSWindow.didChangeOcclusionStateNotification,
    object: window,
    queue: .main
) { _ in
    if !window.occlusionState.contains(.visible) {
        FileHandle.standardError.write(
            "webview-eval: window became occluded; frame timing from here is not trustworthy\n"
                .data(using: .utf8)!
        )
    }
}
_ = occlusionObserver

webView.load(URLRequest(url: url))

DispatchQueue.main.asyncAfter(deadline: .now() + timeout) {
    FileHandle.standardError.write("webview-eval: timed out after \(timeout)s\n".data(using: .utf8)!)
    exit(1)
}

app.run()
