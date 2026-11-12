# `spodeian-web-utils`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.1)-success.svg)]()

**WebAssembly DOM helpers, clipboard manipulation, and client-side blob download triggers.**

---

## Overview

`spodeian-web-utils` abstracts web browser DOM operations with transparent native fallbacks:

- **Client-Side File Downloads**: Browser downloads via Blob URLs on `wasm32`, or a file write on native platforms. `try_trigger_text_download` and `try_trigger_binary_download` return a `Result`; `trigger_text_download` and `trigger_binary_download` log failures instead.
- **Clipboard**: `write_clipboard_text` awaits the browser's Clipboard API and reports whether the write succeeded, so an interface only confirms a copy that happened. Native builds have no clipboard implementation and return `WebUtilsError::Unavailable`; `copy_to_clipboard` is the fire-and-forget form.
