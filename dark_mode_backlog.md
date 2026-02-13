# Dark Mode - Backlog for Future Implementation

**Status**: Backlogged (Feb 13, 2026)
**Reason**: Focus on perfecting light mode first

---

## Current Implementation

The app has full dark mode support built in. Here's what exists:

### Theme Definitions (in qa_tool.py)

```python
# Light Mode (Default)
LIGHT_MODE = {
    'bg': '#FFFFFF',
    'bg_secondary': '#F7F7F7',
    'navbar': '#FFFFFF',
    'card': '#FFFFFF',
    'card_secondary': '#FAFAFA',
    'border': '#E5E5E5',
    'text': '#000000',
    'text_secondary': '#525252',
    'text_muted': '#A3A3A3',
    'accent': '#000000',
    'accent_hover': '#333333',
    'success': '#22C55E',
    'warning': '#F59E0B',
    'error': '#EF4444',
}

# Dark Mode
DARK_MODE = {
    'bg': '#0A0A0A',
    'bg_secondary': '#141414',
    'navbar': '#0A0A0A',
    'card': '#141414',
    'card_secondary': '#1A1A1A',
    'border': '#262626',
    'text': '#FFFFFF',
    'text_secondary': '#A3A3A3',
    'text_muted': '#737373',
    'accent': '#FFFFFF',
    'accent_hover': '#E5E5E5',
    'success': '#22C55E',
    'warning': '#F59E0B',
    'error': '#EF4444',
}
```

### Session State
- `st.session_state.dark_mode` - Boolean flag (True = dark, False = light)
- Default is `False` (light mode)

### Theme Toggle
- Was in navbar as a pill-shaped toggle switch
- Clicking switches between light/dark via URL param `?theme=dark` or `?theme=light`

### get_theme_colors() Function
```python
def get_theme_colors():
    if st.session_state.get('dark_mode', False):
        return DARK_MODE
    return LIGHT_MODE
```

---

## To Re-enable Later

1. Add toggle back to navbar (right side, after BETA badge)
2. Toggle HTML:
```html
<a href="{build_theme_url('light' if is_dark else 'dark')}" target="_parent" style="text-decoration: none; display: block; margin-left: 12px;">
    <div class="proof-toggle" style="background: {theme['bg_secondary']}; border: 1px solid {theme['border']}; width: 48px; height: 26px; border-radius: 13px; position: relative; cursor: pointer;">
        <div style="position: absolute; top: 2px; left: {knob_left}; width: 20px; height: 20px; border-radius: 50%; background: {'#FFFFFF' if is_dark else '#000000'}; transition: left 0.3s ease;"></div>
    </div>
</a>
```

3. `knob_left` variable: `"24px"` if dark, `"2px"` if light

---

## Notes
- All UI components already use `theme['color']` variables
- Switching themes should "just work" once toggle is re-enabled
- Consider adding user preference persistence (save to database)
