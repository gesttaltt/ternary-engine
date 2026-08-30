"""
install_rapl_udev_rule.py - make Intel-RAPL energy counters readable persistently

Copyright 2025 Ternary Engine Contributors
Licensed under the Apache License, Version 2.0

Installs 99-ternary-rapl-readable.rules so that
/sys/class/powercap/intel-rapl:*/energy_uj is readable by the "adm" group
across reboots, which is what commercial-viability criterion 4 (power
consumption) needs in order to be measurable without root.

Those counters ship as 0400 root:root deliberately -- readable RAPL is the
PLATYPUS side-channel surface (CVE-2020-8694/8695). Installing this widens
that to members of "adm". Reasonable on a single-user development machine;
NOT appropriate on a shared or multi-tenant host. See the .rules file for
the full trade-off.

A plain `sudo chmod a+r` does not persist: the powercap devices are
recreated with default permissions on every boot.

USAGE: sudo python3 scripts/setup/install_rapl_udev_rule.py
       python3 scripts/setup/install_rapl_udev_rule.py --check
       sudo python3 scripts/setup/install_rapl_udev_rule.py --uninstall
OUTPUT: installs (or removes) the rule, reloads udev, and VERIFIES the
        resulting permissions rather than assuming the reload worked
"""

import argparse
import grp
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULE_SRC = HERE / "99-ternary-rapl-readable.rules"
RULE_DST = Path("/etc/udev/rules.d/99-ternary-rapl-readable.rules")
GROUP = "adm"
DOMAIN_GLOB = "/sys/class/powercap/intel-rapl:*/energy_uj"


def domains() -> list:
    import glob
    return sorted(glob.glob(DOMAIN_GLOB))


def report() -> bool:
    """Prints current state.

    Returns True only if every domain actually carries the state this rule is
    supposed to produce: group `adm` with group-read set. Deliberately NOT
    os.access(R_OK) -- this script runs under sudo, and root can read a
    0400 root:root file, so an access check would report success even if the
    rule silently did nothing. Checking the mode bits is the only way to tell
    whether udev applied anything.
    """
    found = domains()
    if not found:
        print("  No intel-rapl energy_uj domains found on this machine.")
        print("  RAPL may be unsupported here, or the module is not loaded.")
        return False
    all_ok = True
    for f in found:
        st = os.stat(f)
        mode = st.st_mode & 0o777
        try:
            gname = grp.getgrgid(st.st_gid).gr_name
        except KeyError:
            gname = str(st.st_gid)
        applied = (gname == GROUP) and bool(mode & 0o040)
        # A pre-existing world-readable chmod also leaves it usable, but it is
        # NOT what this rule produces and will not survive a reboot.
        transient = bool(mode & 0o004) and not applied
        all_ok &= applied
        state = "rule applied" if applied else (
            "readable, but via a transient chmod (will not survive reboot)"
            if transient else "NOT readable by group")
        print(f"  {oct(mode)} root:{gname:<6} {state}")
        print(f"      {f}")
    return all_ok


def need_root() -> None:
    if os.geteuid() != 0:
        print("\nThis step needs root. Re-run:")
        print(f"  sudo python3 {Path(__file__).relative_to(Path.cwd()) if str(Path.cwd()) in str(__file__) else __file__}")
        raise SystemExit(1)


def reload_udev() -> None:
    for cmd in (["udevadm", "control", "--reload-rules"],
                ["udevadm", "trigger", "--subsystem-match=powercap"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [WARN] {' '.join(cmd)} failed: {r.stderr.strip()}")
        else:
            print(f"  ran: {' '.join(cmd)}")
    subprocess.run(["udevadm", "settle"], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report current permissions and exit (no root needed)")
    ap.add_argument("--uninstall", action="store_true",
                    help="remove the rule and reload udev")
    args = ap.parse_args()

    print("Intel-RAPL energy counter permissions")
    print("-" * 62)
    print("Before:")
    before_ok = report()

    if args.check:
        installed = RULE_DST.exists()
        print(f"\nRule installed at {RULE_DST}: {'yes' if installed else 'no'}")
        if installed and before_ok:
            print("[OK] persistent across reboots.")
            return 0
        if before_ok:
            print("[OK] rule applied.")
            return 0
        print("[NOT PERSISTENT] run: sudo python3 "
              "scripts/setup/install_rapl_udev_rule.py")
        return 1

    if args.uninstall:
        need_root()
        if RULE_DST.exists():
            RULE_DST.unlink()
            print(f"\nRemoved {RULE_DST}")
        else:
            print(f"\n{RULE_DST} was not present")
        reload_udev()
        print("\nAfter (permissions revert fully on the next reboot):")
        report()
        return 0

    if not RULE_SRC.exists():
        print(f"\n[FAIL] rule file missing: {RULE_SRC}")
        return 1

    try:
        grp.getgrnam(GROUP)
    except KeyError:
        print(f"\n[FAIL] group '{GROUP}' does not exist on this system. Edit "
              f"{RULE_SRC.name} to target a group that does.")
        return 1

    need_root()
    RULE_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RULE_SRC, RULE_DST)
    os.chmod(RULE_DST, 0o644)
    print(f"\nInstalled {RULE_DST}")
    reload_udev()

    print("\nAfter:")
    after_ok = report()

    # Verify rather than assume: udev reload/trigger can succeed while the
    # rule still fails to apply (wrong glob, missing binary, SELinux).
    if not after_ok:
        print("\n[FAIL] the rule did not take effect. Check:")
        print("  udevadm test /sys/class/powercap/intel-rapl:0 2>&1 | tail -20")
        return 1

    invoking = os.environ.get("SUDO_USER")
    if invoking:
        members = set(grp.getgrnam(GROUP).gr_mem)
        if invoking not in members:
            print(f"\n[WARN] user '{invoking}' is not in group '{GROUP}'; the "
                  f"counters are readable by that group but not by them.")
            print(f"  sudo usermod -aG {GROUP} {invoking}   # then log out and back in")
            return 1

    print("\n[OK] RAPL counters are readable and will remain so across reboots.")
    print("     Verify after your next reboot with:")
    print("       python3 scripts/setup/install_rapl_udev_rule.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
