"""Keep existing scientific-export assertions independent of new syntax exports."""
SYNTAX_KEYS = frozenset({'repro_py', 'repro_r', 'repro_zip'})


def scientific_downloads(page):
    # Exclude only the three newly specified syntax controls; all old controls
    # (including any unexpected extra scientific export) remain in the count.
    return [button for button in page.get('download_button') if button.key not in SYNTAX_KEYS]
