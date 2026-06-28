#!/usr/bin/env python3
"""
Colab / Drive checkpoint kalıcılığı — bir kez eğit, sonra sadece yükle.

Kullanım (Colab):
  python3 colab_persist.py save          # eğitim sonrası Drive'a yaz
  python3 colab_persist.py load          # yeni oturumda Drive'dan çek
  python3 colab_persist.py status        # lokal + Drive durumu
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "bulk_swap" / "bulk_v2.pt"
RESULTS = ROOT / "bulk_kvfree_train_results.json"
DRIVE_DIR = Path("/content/drive/MyDrive/infiti")
DRIVE_CKPT = DRIVE_DIR / "bulk_v2.pt"
DRIVE_RESULTS = DRIVE_DIR / "bulk_kvfree_train_results.json"
DRIVE_META = DRIVE_DIR / "infiti_meta.json"


def _mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except ImportError:
        raise SystemExit("Drive mount yalnızca Colab'da çalışır.")


def cmd_save(args):
    if not CKPT.exists():
        raise SystemExit(f"Checkpoint yok: {CKPT}\nÖnce eğitimi bitir.")

    if args.drive:
        _mount_drive()
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CKPT, DRIVE_CKPT)
        if RESULTS.exists():
            shutil.copy2(RESULTS, DRIVE_RESULTS)
        meta = {
            "checkpoint": str(DRIVE_CKPT),
            "size_mb": round(CKPT.stat().st_size / 1e6, 2),
        }
        if RESULTS.exists():
            meta["train_results"] = json.loads(RESULTS.read_text())
        DRIVE_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"✓ Drive'a kaydedildi: {DRIVE_DIR}")
        print(f"  {DRIVE_CKPT} ({meta['size_mb']} MB)")
    else:
        backup = ROOT / "checkpoints" / "bulk_swap" / "bulk_v2_backup.pt"
        shutil.copy2(CKPT, backup)
        print(f"✓ Lokal yedek: {backup}")

    for ep in sorted((ROOT / "checkpoints" / "bulk_swap").glob("bulk_v2_ep*.pt")):
        print(f"  epoch ckpt: {ep.name}")


def cmd_load(args):
    if args.drive:
        _mount_drive()
        if not DRIVE_CKPT.exists():
            raise SystemExit(
                f"Drive'da checkpoint yok: {DRIVE_CKPT}\n"
                "Mac'ten upload veya önce colab_persist.py save çalıştır."
            )
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DRIVE_CKPT, CKPT)
        if DRIVE_RESULTS.exists():
            shutil.copy2(DRIVE_RESULTS, RESULTS)
        print(f"✓ Yüklendi: {CKPT}")
        if DRIVE_META.exists():
            print(DRIVE_META.read_text()[:500])
    else:
        backup = ROOT / "checkpoints" / "bulk_swap" / "bulk_v2_backup.pt"
        if not backup.exists():
            raise SystemExit(f"Yedek yok: {backup}")
        shutil.copy2(backup, CKPT)
        print(f"✓ Lokal yedekten yüklendi: {CKPT}")


def cmd_status(_args):
    print("--- Lokal ---")
    for p in [CKPT, RESULTS]:
        if p.exists():
            print(f"  ✓ {p} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✗ {p}")
    if Path("/content/drive").exists():
        print("--- Drive ---")
        for p in [DRIVE_CKPT, DRIVE_RESULTS, DRIVE_META]:
            if p.exists():
                print(f"  ✓ {p}")
            else:
                print(f"  ✗ {p}")
    else:
        print("--- Drive (mount yok) ---")


def main():
    parser = argparse.ArgumentParser(description="Infiti checkpoint kalıcılığı")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save", help="Eğitim sonrası Drive'a kaydet")
    s.add_argument("--drive", action="store_true", default=True)
    s.add_argument("--local-only", action="store_true")
    s.set_defaults(func=lambda a: cmd_save(type("A", (), {"drive": not a.local_only})()))
    l = sub.add_parser("load", help="Yeni oturumda Drive'dan yükle")
    l.add_argument("--drive", action="store_true", default=True)
    l.add_argument("--local-only", action="store_true")
    l.set_defaults(func=lambda a: cmd_load(type("A", (), {"drive": not a.local_only})()))
    sub.add_parser("status", help="Durum").set_defaults(func=cmd_status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
