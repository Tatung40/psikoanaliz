"""Keep existing scientific-export assertions independent of new syntax exports."""
SYNTAX_KEYS = frozenset({'repro_py', 'repro_zip', 'academic_word'})


def scientific_downloads(page):
    # Exclude the shared syntax/Word controls (tested separately); all old controls
    # (including any unexpected extra scientific export) remain in the count.
    return [button for button in page.get('download_button') if button.key not in SYNTAX_KEYS]
