"""Real-browser verification of held-out/training replays and three pipelines."""

import json

from .common import ROOT, atomic_json, settings, utc_now


def check_demo(url):
    from playwright.sync_api import expect, sync_playwright

    output = ROOT / "artifacts/browser-check"
    output.mkdir(parents=True, exist_ok=True)
    config = settings()
    errors, checks, subjects = [], [], set()
    distribution_checks = []
    role_subjects = {"held_out": set(), "training": set()}
    fits = {key: set() for key in config["comparisons"]}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1200})
        page.add_init_script("""(() => {
            window.traceChannelLabels = {};
            const fillText = CanvasRenderingContext2D.prototype.fillText;
            CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
                if (this.canvas.id.startsWith('traces-') && /^CH \\d+$/.test(text)) {
                    const labels = window.traceChannelLabels[this.canvas.id] ||= new Set();
                    labels.add(Number(text.slice(3)));
                }
                return fillText.call(this, text, ...args);
            };
        })();""")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="domcontentloaded")
        expect(page.locator("#status")).to_contain_text("Ready.", timeout=60000)
        toggle = page.get_by_role("switch", name="Show Prediction Distribution in video")
        expect(toggle).not_to_be_checked()
        expect(page.locator("#video-label-overlay")).to_be_visible()
        expect(page.locator("#video-distribution-overlay")).to_be_hidden()

        def verify_distribution(data, model, index):
            expect(page.locator("#video-label-overlay")).to_be_hidden()
            expect(page.locator("#video-distribution-overlay")).to_be_visible()
            expect(page.locator(".video-distribution-heading")).to_have_text(f"{model['name']} Prediction Distribution")
            probabilities = model["probabilities"][index]
            if data["windows"][index]["mask"] < 0 or probabilities is None:
                expect(page.locator("#video-probabilities")).to_be_visible()
                expect(page.locator("#video-distribution-status")).to_be_hidden()
                rows = page.locator("#video-probabilities .probability")
                expect(rows).to_have_count(len(data["labels"]))
                assert rows.locator(".number").all_text_contents() == ["0.0%"] * len(data["labels"])
                assert rows.locator(".fill").evaluate_all("elements => elements.every(element => parseFloat(element.style.width) === 0)")
                expect(page.locator("#video-probabilities .selected")).to_have_count(0)
                if data["windows"][index]["mask"] < 0:
                    expect(page.locator("#video-distribution-status")).to_have_text("Unscored interval")
                else:
                    expect(page.locator("#video-distribution-status")).to_contain_text("Probability distribution unavailable for SVM")
                distribution_checks.append({"subject": data["subject"], "model": model["id"], "window": index, "zero_placeholder": True})
                return
            expect(page.locator("#video-probabilities")).to_be_visible()
            expect(page.locator("#video-distribution-status")).to_be_hidden()
            rows = page.locator("#video-probabilities .probability")
            expect(rows).to_have_count(len(data["labels"]))
            values = rows.evaluate_all("elements => elements.map(row => ({label:Number(row.dataset.label), selected:row.classList.contains('selected'), width:parseFloat(row.querySelector('.fill').style.width), percent:parseFloat(row.querySelector('.number').textContent)}))")
            maximum = probabilities.index(max(probabilities))
            assert [row["label"] for row in values] == data["labels"]
            assert [row["label"] for row in values if row["selected"]] == [data["labels"][maximum]]
            for row, probability in zip(values, probabilities):
                # CSS percentage serialization rounds to fewer digits than the data.
                assert abs(row["width"] - 100 * probability) < 1e-4
                assert abs(row["percent"] - 100 * probability) <= .051
            side_widths = page.locator("#probabilities .fill").evaluate_all("elements => elements.map(element => parseFloat(element.style.width))")
            assert all(abs(value - 100 * probability) < 1e-4 for value, probability in zip(side_widths, probabilities))
            distribution_checks.append({"subject": data["subject"], "model": model["id"], "window": index, "maximum_label": data["labels"][maximum]})

        recordings = page.request.get(url + "/api/recordings").json()
        expected_roles = {"held_out": set(config["test_subjects"]), "training": set(config.get("demo", {}).get("training_subjects", []))}
        expect(page.locator("#recording optgroup")).to_have_count(2)
        for record in recordings:
            role = record.get("split_role", "held_out")
            assert record["subject"] in expected_roles[role]
            role_subjects[role].add(record["subject"])
            subjects.add(record["subject"])
            previous_toggle = toggle.is_checked()
            page.evaluate("window.traceChannelLabels = {}")
            page.select_option("#recording", record["recording"])
            expect(page.locator("#status")).to_contain_text("Ready.", timeout=60000)
            assert toggle.is_checked() == previous_toggle
            page.wait_for_function("document.getElementById('video').readyState >= 2", timeout=60000)
            crop = page.locator("#video").evaluate("video => {const box=video.getBoundingClientRect(); const style=getComputedStyle(video); const scale=Math.max(box.width/video.videoWidth, box.height/video.videoHeight); return {visibleX:box.width/(scale*video.videoWidth), visibleY:box.height/(scale*video.videoHeight), fit:style.objectFit, position:style.objectPosition, controls:video.controls};}")
            assert abs(crop["visibleX"] - .85) < .001
            assert abs(crop["visibleY"] - 1) < .001
            assert crop["fit"] == "cover" and crop["position"] == "0% 50%"
            assert crop["controls"]
            expect(page.locator("#band")).to_have_count(0)
            expect(page.locator(".trace-band canvas")).to_have_count(3)
            page.wait_for_function("Object.values(window.traceChannelLabels).length === 3 && Object.values(window.traceChannelLabels).every(labels => labels.size === 8)")
            channel_columns = page.evaluate("Object.fromEntries(Object.entries(window.traceChannelLabels).map(([id, labels]) => [id, [...labels].sort((a,b) => a-b)]))")
            assert channel_columns == {
                "traces-elbow": list(range(1, 9)),
                "traces-middle": list(range(9, 17)),
                "traces-wrist": list(range(17, 25)),
            }
            rectangles = page.locator(".trace-band canvas").evaluate_all("elements => elements.map(element => {const r=element.getBoundingClientRect(); return {x:r.x, y:r.y, height:r.height};})")
            assert rectangles[0]["x"] < rectangles[1]["x"] < rectangles[2]["x"]
            assert max(r["y"] for r in rectangles) - min(r["y"] for r in rectangles) < 1
            assert len({r["height"] for r in rectangles}) == 1
            data = page.request.get(url + "/api/replay/" + record["recording"]).json()
            assert data["split_role"] == role
            if role == "training":
                expect(page.locator("#protocol-note")).to_contain_text("In-sample replay")
                expect(page.locator("#accuracy-scope")).to_contain_text("training recording")
            else:
                expect(page.locator("#protocol-note")).to_contain_text("Zero calibration")
                expect(page.locator("#accuracy-scope")).to_contain_text("held-out recording")
            assert {m["id"] for m in data["models"]} == set(config["comparisons"])
            scored = [i for i, w in enumerate(data["windows"]) if w["mask"] >= 0 and w["time"] >= 5]
            for model in data["models"]:
                assert model["train_subjects"] == config["train_subjects"]
                assert model["test_subjects"] == config["test_subjects"]
                assert model["test_calibration_rows"] == 0
                assert model["split_role"] == role
                assert record["subject"] in model["train_subjects" if role == "training" else "test_subjects"]
                fits[model["id"]].add(model["fit_id"])
                page.select_option("#model", model["id"])
                expect(page.locator("#prediction-heading")).to_have_text(f"{model['name']} EMG PREDICTION")
                toggle.uncheck()
                expect(page.locator("#video-label-overlay")).to_be_visible()
                expect(page.locator("#video-distribution-overlay")).to_be_hidden()
                expect(page.locator("#accuracy")).to_have_text(f"{model['metrics']['accuracy'] * 100:.1f}%")
                positions = [scored[0], scored[len(scored) // 2], scored[-1]]
                first_probabilities = model["probabilities"][scored[0]]
                if first_probabilities is not None:
                    first_maximum = first_probabilities.index(max(first_probabilities))
                    changed = next((i for i in scored if model["probabilities"][i].index(max(model["probabilities"][i])) != first_maximum), None)
                    assert changed is not None, "Exercise a change in the most likely gesture"
                    positions.append(changed)
                for index in positions:
                    time = data["windows"][index]["time"] + .01
                    page.locator("#video").evaluate("(v,t) => {v.pause(); v.currentTime=t;}", time)
                    page.wait_for_function("t => !document.getElementById('video').seeking && Math.abs(document.getElementById('video').currentTime-t)<.03", arg=time)
                    expected = data["gesture_names"][data["labels"].index(model["predictions"][index])]
                    expect(page.locator("#prediction")).to_have_text(expected)
                    if not toggle.is_checked():
                        expect(page.locator("#overlay-label")).to_have_text(expected)
                        toggle.check()  # Must switch immediately while the video is paused.
                    verify_distribution(data, model, index)
                    if model["id"] == "svm-rms":
                        expect(page.locator("#confidence")).to_have_text("—")
                    checks.append({"subject": record["subject"], "model": model["id"], "video_seconds": time, "prediction": expected})
                toggle.uncheck()
                expect(page.locator("#video-label-overlay")).to_be_visible()
                expect(page.locator("#overlay-label")).to_have_text(expected)
                expect(page.locator("#video-distribution-overlay")).to_be_hidden()
                toggle.check()
                verify_distribution(data, model, positions[-1])
                scored_box = page.locator("#video-distribution-overlay").bounding_box()
                excluded = next(i for i, w in enumerate(data["windows"]) if w["mask"] < 0 and w["time"] >= 0)
                page.locator("#video").evaluate("(v,t) => {v.currentTime=t;}", data["windows"][excluded]["time"] + .01)
                page.wait_for_function("!document.getElementById('video').seeking")
                verify_distribution(data, model, excluded)
                excluded_box = page.locator("#video-distribution-overlay").bounding_box()
                assert all(abs(scored_box[key] - excluded_box[key]) < 1 for key in ("x", "y", "width", "height")), "Unscored intervals must preserve the distribution layout"
            page.select_option("#model", "tabpfn-du")
            active = next(i for i in scored if data["windows"][i]["label"] == 1)
            page.locator("#video").evaluate("(v,t) => {v.currentTime=t;}", data["windows"][active]["time"] + .01)
            page.wait_for_function("!document.getElementById('video').seeking")
            page.select_option("#speed", "2")
            before = page.locator("#video").evaluate("v => v.currentTime")
            page.locator("#video").evaluate("v => {v.muted=true; return v.play();}")
            page.wait_for_function("t => document.getElementById('video').currentTime > t+.4", arg=before)
            page.locator("#video").evaluate("v => v.pause()")
            assert page.locator("#video").evaluate("v => v.getVideoPlaybackQuality().totalVideoFrames > 0")
            now = page.locator("#video").evaluate("v => v.currentTime")
            current = max(i for i, window in enumerate(data["windows"]) if window["time"] <= now)
            tabpfn = next(model for model in data["models"] if model["id"] == "tabpfn-du")
            verify_distribution(data, tabpfn, current)
            page.screenshot(path=str(output / f"participant-{record['subject']}-distribution.png"), full_page=True)
            toggle.uncheck()
            page.screenshot(path=str(output / f"participant-{record['subject']}.png"), full_page=True)
            toggle.check()
        page.set_viewport_size({"width": 390, "height": 900})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.evaluate("(() => { const panel=document.getElementById('video-distribution-overlay').getBoundingClientRect(); const video=document.querySelector('.video-wrap').getBoundingClientRect(); return panel.left >= video.left && panel.top >= video.top && panel.right <= video.right + 1 && panel.bottom <= video.bottom + 1; })()"), "Distribution overlay must fit inside the mobile video"
        assert page.locator(".trace-scroll").evaluate("element => element.scrollWidth > element.clientWidth")
        page.locator(".trace-scroll").evaluate("element => {element.scrollLeft = element.scrollWidth;}")
        assert page.evaluate("document.getElementById('traces-wrist').getBoundingClientRect().right <= document.querySelector('.trace-scroll').getBoundingClientRect().right + 1")
        page.screenshot(path=str(output / "mobile-wrist.png"), full_page=True)
        assert role_subjects == expected_roles
        assert subjects == expected_roles["held_out"] | expected_roles["training"]
        assert all(len(ids) == 1 for ids in fits.values()), "Each pipeline must share one fit across all users"
        assert not errors, errors
        browser.close()
    result = {
        "status": "passed", "checked_at": utc_now(), "subjects": sorted(subjects),
        "roles": {role: sorted(values) for role, values in role_subjects.items()},
        "shared_fits": {key: list(ids) for key, ids in fits.items()},
        "timestamp_checks": checks, "browser_errors": errors,
        "trace_layout": {"rows": 8, "columns": 3, "channels": channel_columns},
        "distribution_checks": distribution_checks,
    }
    atomic_json(output / "verification.json", result)
    summary = {k: v for k, v in result.items() if k not in {"timestamp_checks", "distribution_checks"}}
    summary["timestamp_checks_passed"] = len(checks)
    summary["distribution_checks_passed"] = len(distribution_checks)
    print(json.dumps(summary, indent=2))
    return result
