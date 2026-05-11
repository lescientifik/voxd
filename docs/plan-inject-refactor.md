---
description: Refactor de l'injection clavier voxd — paste via raccourci au lieu de typing char-by-char, avec détection de la fenêtre focus pour choisir Ctrl+V vs Ctrl+Shift+V, notif sur Xwayland.
---

# Plan inject refactor — voxd

## État

Le code v0.1.0 fait `wl-copy` + `wtype -- <text>` (typing char-by-char). Deux problèmes UX en prod :

1. **Lenteur** sur les longs textes (chaque char = 1 keystroke).
2. **Chars perdus, dont des espaces** — bug upstream confirmé ([atx/wtype#46](https://github.com/atx/wtype/issues/46)), repo abandonné depuis 2021, pas de fix mergé. Causes multiples : allocation keycode, lifecycle virtual-keyboard, absence de flow control Wayland.

Direction validée : **paste via Ctrl+V** (clipboard déjà setté par `wl-copy`, on envoie juste la combo). Plus de typing char-by-char nulle part dans le nouveau code.

## Architecture cible

Le pipeline post-refactor, exécuté dans `inject()` (sync, déjà off-loop via `run_in_executor`) :

```
inject(text, cfg)
 ├─ wl-copy <text>           (toujours, si cfg.clipboard)
 └─ if cfg.type:
     │
     ├─ detect_focus() via `swaymsg -t get_tree`
     │
     ├─ Wayland-native + app_id ∈ TERMINAL_SET
     │     → wtype -M ctrl -M shift -k v -m shift -m ctrl
     │
     ├─ Wayland-native + app_id ∉ TERMINAL_SET
     │     → wtype -M ctrl -k v -m ctrl
     │
     ├─ Xwayland (app_id null, window_properties non-null)
     │     → notify-send "voxd: paste manually (XWayland app)"
     │       (pas de wtype — wtype ne peut pas frapper Xwayland, atx/wtype#62)
     │
     └─ Unknown (swaymsg absent / parse fail / no focused container)
           → wtype -M ctrl -k v -m ctrl   (best-effort, marche sur Hyprland/labwc)
```

`TERMINAL_SET` est hardcodée dans `src/voxd/inject.py` (frozenset module-level) :

```python
_TERMINAL_APP_IDS = frozenset({
    "foot",
    "Alacritty",              # casing exact, A capital
    "kitty",
    "org.wezfurlong.wezterm",
    "com.mitchellh.ghostty",
})
```

Validation supplémentaire au boot du daemon : `inject.type=true && inject.clipboard=false` est rejeté (paste-via-Ctrl+V a besoin que le clipboard soit setté ; sans ça, on collerait le contenu précédent du clipboard — pire qu'un no-op).

## Décisions verrouillées

| # | Décision | Justification |
|---|----------|---------------|
| D1 | Paste via Ctrl+V, jamais de typing char-by-char | Bug wtype #46 non corrigeable, repo abandonné |
| D2 | Détection focus via `swaymsg -t get_tree` | Pas de dép externe, JSON stable, déjà installé sur cible |
| D3 | Liste terminaux **hardcodée** dans `inject.py` (5 entrées) | Liste petite et stable, YAGNI pour la config TOML |
| D4 | Xwayland focusé → notif + skip wtype | wtype #62 — wtype ne peut pas frapper Xwayland |
| D5 | Focus indéterminé → fallback Ctrl+V brut | Conservateur, supporte Hyprland/labwc/edge cases sway |
| D6 | `type=true && clipboard=false` → daemon refuse de démarrer | Config incohérente post-refactor, exit propre > UX mystérieuse |
| D7 | Flag `inject.type` conservé tel quel | Kill switch d'injection, sémantique inchangée du point de vue user |
| D8 | Détection focus sync dans `inject()` | inject() tourne déjà via `run_in_executor`, pas de plomberie async supplémentaire |
| D9 | `wl-copy` passe de "warning" à "critical" dans `voxd doctor` | Sans clipboard, Ctrl+V colle le mauvais texte → UX cassée |
| D10 | `notify-send` reste "warning" dans doctor | La notif Xwayland est nice-to-have, voxd marche sans |

## Faits techniques de référence

- **Syntaxe wtype combos** (`man wtype`) : `-M MOD` press, `-m MOD` release, `-k KEY` type (press+release). Modifiers valides : `shift, capslock, ctrl, logo, win, alt, altgr`.
- **wtype + layout** : `-k v` envoie la *keysym* `v` via libxkbcommon (nom logique, pas position physique). Marche sur AZERTY/QWERTY/etc.
- **sway tree shape (vérifié)** : nodes ont `focused: bool`, `app_id: str | null` (top-level, Wayland-native uniquement), `window_properties: {class, instance, ...} | null` (Xwayland uniquement). Le node container fenêtre a `type: "con"` (workspaces et root sont d'autres types).
- **wtype + Xwayland** : ne marche pas du tout ([atx/wtype#62](https://github.com/atx/wtype/issues/62)).

## Méthodologie

Identique au plan v0.1.0 (voir [plan-voxd.md](plan-voxd.md) section "Méthodologie d'exécution") : TDD red/green, **un subagent Opus par step en séquentiel**, un commit conventional par step. Briefing-type du subagent : autonomie maximale sur structure interne, nommage privé, factorisation ; aucune autorité pour modifier les décisions verrouillées ci-dessus.

## Roadmap TDD

### Step 1 — Focus detector (`src/voxd/focus.py`)

Module standalone : appelle `swaymsg -t get_tree`, parse le JSON, retourne un verdict typé. Pas de connaissance des combos wtype ni de la TERMINAL_SET — c'est de la pure plomberie sway.

**Délivrable** :

```python
# src/voxd/focus.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class FocusKind(Enum):
    WAYLAND = "wayland"      # app_id présent, window_properties null
    XWAYLAND = "xwayland"    # app_id null, window_properties présent (class connue)
    UNKNOWN = "unknown"      # swaymsg absent, parse fail, ou aucun container focused

@dataclass(frozen=True)
class FocusInfo:
    kind: FocusKind
    app_id: str | None = None    # set ssi kind == WAYLAND

def detect_focus() -> FocusInfo:
    """Retourne le focus courant en interrogeant `swaymsg -t get_tree`.

    Tous les modes d'échec (swaymsg manquant, exit non-zéro, JSON invalide,
    aucun container focused) collapsent sur FocusKind.UNKNOWN.
    """
```

**Tests rouges** (`tests/test_focus.py`, nouveau fichier) :

- `test_detect_focus_wayland_native` : monkeypatch `subprocess.run` pour retourner un JSON sway minimal avec un node `{type: "con", focused: true, app_id: "foot", window_properties: null}` → `FocusInfo(WAYLAND, "foot")`.
- `test_detect_focus_xwayland` : node `{type: "con", focused: true, app_id: null, window_properties: {"class": "XTerm"}}` → `FocusInfo(XWAYLAND, None)`.
- `test_detect_focus_no_focused_container` : tree sans aucun node `focused: true` parmi les containers → `FocusInfo(UNKNOWN, None)`.
- `test_detect_focus_swaymsg_missing` : `subprocess.run` lève `FileNotFoundError` → `FocusInfo(UNKNOWN, None)`.
- `test_detect_focus_swaymsg_nonzero` : `subprocess.run` retourne un `CompletedProcess` avec returncode != 0 → `FocusInfo(UNKNOWN, None)`.
- `test_detect_focus_invalid_json` : stdout = `"not json"` → `FocusInfo(UNKNOWN, None)`.
- `test_detect_focus_nested_tree` : focused container est profondément nesté (root > output > workspace > container > **focused leaf**) — confirme que le walk est récursif.
- `test_detect_focus_ignores_focused_workspace` : un node `type: "workspace", focused: true` sans app_id ne compte pas (seuls les `type: "con"` qualifient).

**DoD** : tests verts, `ruff check` + `ty check` clean. Pas d'usage outside `focus.py` (utilisé seulement au step 2).

**Commit** : `feat(focus): swaymsg-based sway focus detector with fallbacks`

---

### Step 2 — Refactor `inject.py` (paste-via-Ctrl+V)

Réécrit `inject()` pour utiliser `detect_focus()` et router vers une des 4 branches. Supprime intégralement l'ancien `_run_wtype(text)` (typing char-by-char). Ajoute la helper `_run_notify_send(message)` pour la branche Xwayland.

**Délivrable** : `src/voxd/inject.py` réécrit. Squelette :

```python
_TERMINAL_APP_IDS = frozenset({
    "foot", "Alacritty", "kitty",
    "org.wezfurlong.wezterm", "com.mitchellh.ghostty",
})

def _run_wtype_combo(modifiers: list[str], key: str) -> None:
    """wtype -M m1 -M m2 ... -k key -m m2 -m m1 (release reverse order)."""

def _run_notify_send(summary: str, body: str) -> None:
    """Best-effort notify-send call; swallow failures (it's informational)."""

def inject(text: str, cfg: InjectConfig) -> None:
    if not text:
        return
    if cfg.clipboard:
        _run_wlcopy(text)
    if cfg.type:
        focus = detect_focus()
        if focus.kind == FocusKind.XWAYLAND:
            _run_notify_send("voxd", "Paste manually — XWayland app detected (Ctrl+V or Ctrl+Shift+V)")
            return
        if focus.kind == FocusKind.WAYLAND and focus.app_id in _TERMINAL_APP_IDS:
            _run_wtype_combo(["ctrl", "shift"], "v")
        else:
            _run_wtype_combo(["ctrl"], "v")
```

**Tests rouges** (`tests/test_inject.py`, partiellement réécrit) :

Tests existants à *retirer* (le contrat a changé) :
- `test_inject_default_calls_both_wlcopy_and_wtype` (l'ancien assertait `wtype -- <text>`)
- `test_inject_utf8_french_accents` côté wtype (le texte n'est plus passé à wtype)

Tests existants à *garder tels quels* :
- `test_inject_clipboard_false_skips_wlcopy`
- `test_inject_empty_text_is_noop`
- `test_inject_wlcopy_called_before_wtype` (l'ordre wl-copy → wtype reste valable)
- `test_inject_propagates_subprocess_error` (wrap toujours en `InjectError`)
- `test_inject_called_process_error_wrapped`

Nouveaux tests :
- `test_inject_wayland_non_terminal_uses_ctrl_v` : monkeypatch `detect_focus` → `WAYLAND/"google-chrome"` → argv wtype contient `["-M", "ctrl", "-k", "v", "-m", "ctrl"]`, *pas* de `-M shift`.
- `test_inject_wayland_terminal_uses_ctrl_shift_v` : pour chaque app_id de la TERMINAL_SET, focus → argv wtype contient `-M ctrl -M shift -k v -m shift -m ctrl`.
- `test_inject_xwayland_skips_wtype_and_notifies` : focus → `XWAYLAND` → `subprocess.run` est appelé pour `notify-send` mais **pas** pour `wtype`. Le message notify contient "paste manually".
- `test_inject_unknown_focus_uses_ctrl_v` : focus → `UNKNOWN` → argv wtype = Ctrl+V brut (comme non-terminal Wayland).
- `test_inject_xwayland_still_runs_wlcopy_first` : clipboard est setté même si on skip wtype.
- `test_inject_notify_send_failure_is_swallowed` : si `notify-send` lève FileNotFoundError, `inject()` n'élève pas (la notif est best-effort, on a déjà le clipboard).
- `test_inject_type_false_skips_focus_detection` : avec `type=False`, `detect_focus` n'est **pas** appelé (perf + pas de subprocess inutile).
- `test_inject_wtype_combo_release_modifiers_reverse_order` : la release suit l'ordre inverse de la press (`-M ctrl -M shift -k v -m shift -m ctrl` et non `-m ctrl -m shift`). Pas une exigence wtype stricte, mais c'est l'idiome du README upstream.
- `test_inject_utf8_text_still_reaches_wlcopy` : confirme que `wl-copy` reçoit les accents tels quels (le seul chemin qui voit encore le texte côté inject).

**DoD** :
- Tests verts (anciens supprimés + nouveaux).
- `_run_wtype(text)` (l'ancien) n'existe plus dans le module.
- `ruff` + `ty` clean.
- Le module n'expose toujours que `inject` et `InjectError` (pas de changement d'API publique).

**Commit** : `refactor(inject): paste via Ctrl+V instead of char-by-char typing`

---

### Step 3 — Validation config au boot du daemon

Rejet de `inject.type=true && inject.clipboard=false` au démarrage du daemon. Validation dans `Daemon.__init__` (la config reste un POPO chargeable sans validation, le daemon en applique la sémantique).

**Délivrable** : ajouter dans `Daemon.__init__` un check après l'assignation `self._cfg = cfg` :

```python
if cfg.inject.type and not cfg.inject.clipboard:
    raise ValueError(
        "invalid config: inject.type=true requires inject.clipboard=true "
        "(paste-via-keystroke needs the clipboard to be set). "
        "Either set clipboard=true, or set type=false."
    )
```

Et dans `voxd/cli.py` (point d'entrée du daemon foreground) : catcher cette `ValueError` au démarrage, l'afficher proprement sur stderr, et exit code 2 (= config invalide, distinct du 1 = runtime error).

**Tests rouges** :

- `tests/test_daemon_e2e.py` : `test_daemon_rejects_type_true_clipboard_false_at_init` — construire un Daemon avec `InjectConfig(type=True, clipboard=False)` → `ValueError` levée avec message contenant `"clipboard"`.
- `tests/test_daemon_e2e.py` : `test_daemon_accepts_type_false_clipboard_false` — `InjectConfig(type=False, clipboard=False)` est OK (daemon démarre, transcrit, fait rien) ; pas d'erreur de validation même si cette config rend voxd inutile.
- `tests/test_daemon_e2e.py` : `test_daemon_accepts_type_true_clipboard_true` — config par défaut accepte.
- `tests/test_cli.py` : `test_cli_invalid_inject_config_exits_2_with_message` — patcher `Config.load` pour renvoyer une config invalide, lancer le CLI, vérifier `exit code 2` et stderr contenant `"clipboard"` (et idéalement le nom du fichier de config).

**DoD** : tests verts. L'erreur est attrapée en deux endroits cohérents : `Daemon.__init__` (raise) + `cli` (catch + exit propre).

**Commit** : `feat(daemon): reject inject.type=true with clipboard=false at boot`

---

### Step 4 — Doctor + README polish

`wl-copy` devient critical dans `voxd doctor` (sans lui, paste-via-Ctrl+V colle le mauvais texte). README updaté pour refléter la nouvelle UX (paste-via-keystroke + comportement Xwayland + comportement hors-sway).

**Délivrable** :
- `src/voxd/doctor.py` : passer `wl-copy` à `critical=True` et reformuler son `what_for` (de "clipboard mirroring will be disabled" à "required for paste-via-Ctrl+V to work").
- `README.md` : section "How injection works" qui décrit les 4 cas (terminal, autre app, Xwayland, autre comp) en 4 bullets. Lien vers `atx/wtype#62` pour la limitation Xwayland.

**Tests rouges** :

- `tests/test_doctor.py` : `test_wlcopy_missing_is_critical` — monkeypatch `shutil.which("wl-copy")` → `None` → `_run_checks()` retourne au moins une `_Check` avec `name="wl-copy"`, `critical=True`, et `main()` retourne 1.
- Le test existant `test_wlcopy_missing_is_not_critical` (ou équivalent) est *supprimé* (le contrat a changé).

**DoD** : tests verts. README à jour (revue manuelle), doctor sort le bon exit code selon `wl-copy`.

**Commit** : `chore(doctor,docs): make wl-copy critical, document paste-via-Ctrl+V flows`

---

## Risques connus / hors scope

- **Bug Chromium/Electron sur wtype** (issue #74 jamais mergée) : on envoie *seulement* Ctrl+V, pas du texte brut. La collision keycode `-M`/`-P` ne nous concerne donc plus. Si dans le futur un combo touchant ces lettres devenait nécessaire (improbable), il faudrait ré-examiner.
- **Apps Wayland-native qui n'acceptent pas Ctrl+V** : extrêmement rare en pratique. Si un user le rencontre, la mitigation est `inject.type = false` (clipboard only) + paste manuel.
- **`swaymsg` lent** : sur charge CPU élevée le subprocess peut prendre 100-200 ms. Comme inject() tourne déjà via `run_in_executor`, le worker upload n'est pas bloqué — seule la latence perçue de l'injection augmente. Pas de mitigation prévue, à mesurer si signalé.

## Références

- Plan v0.1.0 (mère du projet, méthodo) : [plan-voxd.md](plan-voxd.md)
- Code inject actuel : `src/voxd/inject.py`
- Tests inject actuels : `tests/test_inject.py`
- Config schema : `src/voxd/config.py` (`InjectConfig`)
- Call-site dans le daemon : `src/voxd/daemon.py` (`_inject_text`, ligne ~413)
- Issues upstream : [atx/wtype#46](https://github.com/atx/wtype/issues/46), [atx/wtype#62](https://github.com/atx/wtype/issues/62)
