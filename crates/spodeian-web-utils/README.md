# `spodeian-web-utils`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**WebAssembly DOM helpers, clipboard manipulation, and client-side blob download triggers.**

---

## Overview

`spodeian-web-utils` abstracts web browser DOM operations with transparent native fallbacks:

- **Client-Side File Downloads**: Triggers instant browser downloads (`trigger_text_download`, `trigger_binary_download`) via synthetic `<a>` tags and Blob URLs on `wasm32`, writing directly to the filesystem on native platforms.
- **Clipboard Management**: Seamlessly copies text to the browser clipboard using the asynchronous Clipboard API with native fallbacks.
- **Runtime Environment Detection**: Clean detection of web browser vs native desktop execution.
