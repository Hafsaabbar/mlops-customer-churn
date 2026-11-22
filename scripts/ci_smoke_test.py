#!/usr/bin/env python3
"""
CI smoke test script for Customer Churn Prediction REST API.
Validates nominal and degraded container states.
Only Python standard library is used.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

VALID_PAYLOAD = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def load_model_metadata():
    """Load model metadata from host artifacts/model_metadata.json."""
    candidates = [
        pathlib.Path("artifacts/model_metadata.json"),
        pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "model_metadata.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Could not find artifacts/model_metadata.json")


def run_command(cmd, timeout=30):
    """Run a shell/system command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def check_container_health(container, target_status, timeout_seconds):
    """Poll container health until target status ('healthy' or 'unhealthy') is reached."""
    start_time = time.time()
    last_status = "unknown"
    while time.time() - start_time < timeout_seconds:
        rc, out, _ = run_command(f'docker inspect --format "{{{{.State.Health.Status}}}}" {container}')
        if rc == 0 and out:
            last_status = out
            if last_status == target_status:
                elapsed = round(time.time() - start_time, 1)
                return True, f"Container reached '{target_status}' state in {elapsed}s"
            # Early break if container has reached the opposite terminal state
            if target_status == "healthy" and last_status == "unhealthy":
                return False, f"Container transitioned to terminal 'unhealthy' state (expected '{target_status}')"
            if target_status == "unhealthy" and last_status == "healthy":
                return False, f"Container transitioned to terminal 'healthy' state (expected '{target_status}')"
        time.sleep(2)
    return False, f"Timeout after {timeout_seconds}s waiting for '{target_status}'. Last status: '{last_status}'"


def http_get(url, timeout=10):
    """Perform HTTP GET, returning (status_code, body_text)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def http_post(url, payload, timeout=10):
    """Perform HTTP POST with JSON body, returning (status_code, body_text)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        return e.code, body
    except Exception as e:
        return 0, str(e)


def compute_file_sha256(file_path):
    """Compute sha256 of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def run_nominal_checks(base_url, container, image, commit, timeout, metadata):
    """Execute all nominal checks."""
    checks = []
    expected_threshold = float(metadata.get("optimal_threshold", 0.0))
    expected_run_id = str(metadata.get("run_id", ""))

    # 1. Container health
    ok, detail = check_container_health(container, "healthy", timeout)
    checks.append({"name": "container_health_status", "ok": ok, "detail": detail})

    # 2. GET /health
    code, body = http_get(f"{base_url}/health")
    if code == 200:
        try:
            data = json.loads(body)
            if data.get("status") == "ok":
                checks.append({"name": "get_health_nominal", "ok": True, "detail": f"Status 200, body: {data}"})
            else:
                checks.append({"name": "get_health_nominal", "ok": False, "detail": f"Status 200 but unexpected body: {body}"})
        except Exception as e:
            checks.append({"name": "get_health_nominal", "ok": False, "detail": f"JSON parse error: {e}"})
    else:
        checks.append({"name": "get_health_nominal", "ok": False, "detail": f"Expected HTTP 200, got {code}. Body: {body}"})

    # 3. GET /model-info
    code, body = http_get(f"{base_url}/model-info")
    if code == 200:
        try:
            info = json.loads(body)
            m_thresh = info.get("threshold") == expected_threshold
            m_run = info.get("run_id") == expected_run_id
            m_reg = info.get("registry_checked") is False
            m_src = info.get("source") == "joblib"
            if m_thresh and m_run and m_reg and m_src:
                checks.append({
                    "name": "get_model_info_nominal",
                    "ok": True,
                    "detail": f"threshold={info.get('threshold')}, run_id={info.get('run_id')}, registry_checked=False, source=joblib",
                })
            else:
                checks.append({
                    "name": "get_model_info_nominal",
                    "ok": False,
                    "detail": f"Mismatch in model-info: threshold_match={m_thresh}, run_id_match={m_run}, registry_checked_match={m_reg}, source_match={m_src}. Body: {body}",
                })
        except Exception as e:
            checks.append({"name": "get_model_info_nominal", "ok": False, "detail": f"JSON parse error: {e}"})
    else:
        checks.append({"name": "get_model_info_nominal", "ok": False, "detail": f"Expected HTTP 200, got {code}. Body: {body}"})

    # 4. POST /predict (valid)
    code, body = http_post(f"{base_url}/predict", VALID_PAYLOAD)
    if code == 200:
        try:
            pred = json.loads(body)
            expected_keys = {"churn_probability", "prediction", "threshold", "model_version", "model_source"}
            has_keys = set(pred.keys()) == expected_keys
            prob = pred.get("churn_probability")
            prob_valid = isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0
            prediction = pred.get("prediction")
            decision_valid = prediction in {"Yes", "No"}
            expected_decision = "Yes" if prob >= (expected_threshold - 0.0001) else "No"
            decision_match = (prediction == expected_decision)

            if has_keys and prob_valid and decision_valid and decision_match:
                checks.append({
                    "name": "post_predict_valid_nominal",
                    "ok": True,
                    "detail": f"Status 200, prob={prob}, prediction={prediction}, threshold={pred.get('threshold')}",
                })
            else:
                checks.append({
                    "name": "post_predict_valid_nominal",
                    "ok": False,
                    "detail": f"Invalid predict response: has_keys={has_keys}, prob_valid={prob_valid}, decision_match={decision_match}. Body: {body}",
                })
        except Exception as e:
            checks.append({"name": "post_predict_valid_nominal", "ok": False, "detail": f"JSON parse error: {e}"})
    else:
        checks.append({"name": "post_predict_valid_nominal", "ok": False, "detail": f"Expected HTTP 200, got {code}. Body: {body}"})

    # 5. POST /predict (invalid: Contract='Lifetime')
    invalid_payload = dict(VALID_PAYLOAD)
    invalid_payload["Contract"] = "Lifetime"
    code, body = http_post(f"{base_url}/predict", invalid_payload)
    if code == 422:
        try:
            err_data = json.loads(body)
            has_no_pred = "prediction" not in err_data
            if has_no_pred:
                checks.append({"name": "post_predict_invalid_422", "ok": True, "detail": "Status 422 as expected, no 'prediction' key"})
            else:
                checks.append({"name": "post_predict_invalid_422", "ok": False, "detail": f"Status 422 but contains 'prediction' key: {body}"})
        except Exception:
            checks.append({"name": "post_predict_invalid_422", "ok": True, "detail": "Status 422 as expected (non-JSON body)"})
    else:
        checks.append({"name": "post_predict_invalid_422", "ok": False, "detail": f"Expected HTTP 422, got {code}. Body: {body}"})

    # 6. GET /metrics
    code, body = http_get(f"{base_url}/metrics")
    if code == 200:
        has_counter = "predictions_total" in body
        has_ready = "model_ready 1.0" in body
        if has_counter and has_ready:
            checks.append({"name": "get_metrics_nominal", "ok": True, "detail": "Contains 'predictions_total' and 'model_ready 1.0'"})
        else:
            checks.append({"name": "get_metrics_nominal", "ok": False, "detail": f"Metrics check failed: predictions_total={has_counter}, model_ready={has_ready}"})
    else:
        checks.append({"name": "get_metrics_nominal", "ok": False, "detail": f"Expected HTTP 200, got {code}"})

    # 7. Image and container inspection checks
    if image:
        # 7a. Revision label
        rc, out, _ = run_command(f'docker inspect --format "{{{{index .Config.Labels \\"org.opencontainers.image.revision\\"}}}}" {image}')
        rev_match = (rc == 0 and out.strip() == commit.strip())
        checks.append({
            "name": "image_label_revision",
            "ok": rev_match,
            "detail": f"Expected revision='{commit}', got='{out}' (rc={rc})",
        })

        # 7b. Run id label
        rc, out, _ = run_command(f'docker inspect --format "{{{{index .Config.Labels \\"churn.model.run_id\\"}}}}" {image}')
        run_match = (rc == 0 and out.strip() == expected_run_id.strip())
        checks.append({
            "name": "image_label_run_id",
            "ok": run_match,
            "detail": f"Expected run_id='{expected_run_id}', got='{out}' (rc={rc})",
        })

        # 7c. Absence of mlflow and pytest
        cmd = f'docker run --rm --entrypoint python {image} -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(\'mlflow\') is None and importlib.util.find_spec(\'pytest\') is None else 1)"'
        rc, _, err = run_command(cmd)
        checks.append({
            "name": "image_absence_mlflow_pytest",
            "ok": (rc == 0),
            "detail": "mlflow and pytest absent in image" if rc == 0 else f"Check failed (rc={rc}, err={err})",
        })

        # 7d. sha256 of champion_model.joblib
        host_joblib = pathlib.Path("artifacts/champion_model.joblib")
        if not host_joblib.exists():
            host_joblib = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "champion_model.joblib"
        host_sha = compute_file_sha256(host_joblib)
        rc, out, err = run_command(f'docker run --rm --entrypoint sha256sum {image} /app/artifacts/champion_model.joblib')
        if rc != 0:
            rc, out, err = run_command(f'docker run --rm {image} sha256sum artifacts/champion_model.joblib')
        image_sha = out.split()[0].lower() if out else ""
        sha_match = (rc == 0 and image_sha == host_sha)
        checks.append({
            "name": "image_joblib_sha256",
            "ok": sha_match,
            "detail": f"Host SHA={host_sha[:12]}..., Image SHA={image_sha[:12]}...",
        })

        # 7e. Non-root user in container
        rc, out, _ = run_command(f"docker exec {container} id -u")
        non_root = (rc == 0 and out.strip() != "0")
        checks.append({
            "name": "container_non_root_user",
            "ok": non_root,
            "detail": f"Container runs as uid={out} (non-root)" if non_root else f"Container runs as root or command failed: uid={out}",
        })

    return checks


def run_degraded_checks(base_url, container, timeout):
    """Execute all degraded checks."""
    checks = []

    # 1. Container health (expected 'unhealthy')
    ok, detail = check_container_health(container, "unhealthy", timeout)
    checks.append({"name": "container_health_degraded", "ok": ok, "detail": detail})

    # 2. GET /health (expected 503)
    code, body = http_get(f"{base_url}/health")
    checks.append({
        "name": "get_health_degraded_503",
        "ok": (code == 503),
        "detail": f"Status={code} (expected 503), body={body}",
    })

    # 3. POST /predict (expected 503 without 'prediction' key)
    code, body = http_post(f"{base_url}/predict", VALID_PAYLOAD)
    has_no_pred = True
    try:
        data = json.loads(body)
        has_no_pred = "prediction" not in data
    except Exception:
        pass
    pred_ok = (code == 503 and has_no_pred)
    checks.append({
        "name": "post_predict_degraded_503",
        "ok": pred_ok,
        "detail": f"Status={code} (expected 503), no_prediction_key={has_no_pred}",
    })

    # 4. GET /metrics (contains 'model_ready 0.0')
    code, body = http_get(f"{base_url}/metrics")
    has_zero_ready = "model_ready 0.0" in body
    checks.append({
        "name": "get_metrics_degraded",
        "ok": (code == 200 and has_zero_ready),
        "detail": f"Status={code}, contains 'model_ready 0.0': {has_zero_ready}",
    })

    return checks


def main():
    parser = argparse.ArgumentParser(description="CI Smoke Test for Customer Churn API")
    parser.add_argument("mode", choices=["nominal", "degraded"], help="Execution mode")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of API service")
    parser.add_argument("--container", required=True, help="Container name to inspect")
    parser.add_argument("--image", default="", help="Docker image name (required in nominal mode)")
    parser.add_argument("--commit", default="", help="Git commit revision (required in nominal mode)")
    parser.add_argument("--report", default="", help="Path to write JSON test report")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for healthcheck")

    args = parser.parse_args()

    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()

    metadata = {}
    try:
        metadata = load_model_metadata()
    except Exception as e:
        print(f"Warning: Failed to load host metadata: {e}")

    if args.mode == "nominal":
        checks = run_nominal_checks(
            base_url=args.base_url.rstrip("/"),
            container=args.container,
            image=args.image,
            commit=args.commit,
            timeout=args.timeout,
            metadata=metadata,
        )
    else:
        checks = run_degraded_checks(
            base_url=args.base_url.rstrip("/"),
            container=args.container,
            timeout=args.timeout,
        )

    t1 = time.time()
    ended_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    duration = round(t1 - t0, 2)

    all_ok = all(c["ok"] for c in checks)
    status_str = "PASS" if all_ok else "FAIL"

    # Print summary table
    print("=" * 80)
    print(f"CI SMOKE TEST REPORT | Mode: {args.mode.upper()} | Container: {args.container} | Status: {status_str}")
    print("=" * 80)
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark:4s}] {c['name']:30s} : {c['detail']}")
    print("-" * 80)
    passed_count = sum(1 for c in checks if c["ok"])
    print(f"Result: {passed_count}/{len(checks)} checks passed in {duration}s")
    print("=" * 80)

    # Save report if requested
    report_data = {
        "mode": args.mode,
        "status": status_str,
        "container": args.container,
        "image": args.image,
        "commit": args.commit,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "duration_seconds": duration,
        "checks": checks,
    }

    if args.report:
        report_path = pathlib.Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Report saved to: {report_path.resolve()}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
