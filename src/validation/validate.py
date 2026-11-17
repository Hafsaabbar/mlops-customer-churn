"""
Data validation runner using Pandera schemas.
Validates train, test, and stream splits, generating validation reports.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from src.validation.schema import feature_schema, target_schema

def validate_dataframe(df_features: pd.DataFrame, df_target: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Validate features and optional target using Pandera schemas.
    Returns dictionary with validation status, check counts, and any errors.
    """
    errors = []
    is_valid = True

    try:
        feature_schema.validate(df_features, lazy=True)
    except SchemaErrors as err:
        is_valid = False
        failure_cases = err.failure_cases.to_dict(orient="records")
        for fc in failure_cases:
            errors.append({
                "component": "features",
                "column": str(fc.get("column")),
                "check": str(fc.get("check")),
                "failure_case": str(fc.get("failure_case")),
                "index": str(fc.get("index"))
            })
    except Exception as exc:
        is_valid = False
        errors.append({"component": "features", "general_error": str(exc)})

    if df_target is not None:
        try:
            target_schema.validate(df_target, lazy=True)
        except SchemaErrors as err:
            is_valid = False
            failure_cases = err.failure_cases.to_dict(orient="records")
            for fc in failure_cases:
                errors.append({
                    "component": "target",
                    "column": str(fc.get("column")),
                    "check": str(fc.get("check")),
                    "failure_case": str(fc.get("failure_case")),
                    "index": str(fc.get("index"))
                })
        except Exception as exc:
            is_valid = False
            errors.append({"component": "target", "general_error": str(exc)})

    return {
        "valid": is_valid,
        "n_rows": len(df_features),
        "n_columns": len(df_features.columns),
        "error_count": len(errors),
        "errors": errors
    }

def validate_all_splits(splits_dir: Path = Path("data/splits"), report_dir: Path = Path("reports/validation")) -> Dict[str, Any]:
    """Validate all three dataset splits and export a formal report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    all_passed = True

    for split in ["train", "test", "stream"]:
        x_path = splits_dir / split / f"X_{split}.csv"
        y_path = splits_dir / split / f"y_{split}.csv"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing split files for {split} at {splits_dir}")

        df_x = pd.read_csv(x_path)
        df_y = pd.read_csv(y_path)

        res = validate_dataframe(df_x, df_y)
        results[split] = res
        if not res["valid"]:
            all_passed = False

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_valid": all_passed,
        "splits_validated": results
    }

    # Save JSON report
    json_path = report_dir / "validation_report.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save Markdown report
    md_lines = [
        "# Rapport de Validation des Données (Pandera)",
        "",
        f"- **Date d'exécution** : {summary['timestamp_utc']}",
        f"- **Statut Global** : {'VALIDE (SUCCÈS)' if all_passed else 'NON CONFORME (ÉCHEC)'}",
        "",
        "## Résultats par Split",
        "",
        "| Split | Statut | Lignes | Colonnes | Erreurs détectées |",
        "|---|---|---|---|---|"
    ]

    for split, r in results.items():
        st = "✅ Valide" if r["valid"] else "❌ Invalide"
        md_lines.append(f"| `{split}` | {st} | {r['n_rows']} | {r['n_columns']} | {r['error_count']} |")

    md_lines.append("")
    if not all_passed:
        md_lines.append("## Détail des Erreurs")
        for split, r in results.items():
            if not r["valid"]:
                md_lines.append(f"### Split `{split}`")
                for e in r["errors"]:
                    md_lines.append(f"- **{e.get('column')}** ({e.get('check')}): valeur '{e.get('failure_case')}'")

    md_path = report_dir / "validation_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Validation completed. Overall valid: {all_passed}")
    print(f"Reports saved to {json_path} and {md_path}")
    return summary

if __name__ == "__main__":
    from typing import Dict, Any
    validate_all_splits()
