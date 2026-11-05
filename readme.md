# Pipeline: Install and Set Up VS Code + Python + Essential Extensions

This guide provides a reproducible setup pipeline for Windows, macOS, and Linux using command-based steps.

## Tutorial em formato de website

Abra o [tutorial completo em formato de website](tutorial.html), que reúne os conteúdos da pasta `cursos/`, incluindo os capítulos em notebook e os materiais em Markdown.

Quando o GitHub Pages estiver ativado para este repositório, o tutorial também poderá ser acessado pelo endereço:

`https://SEU_USUARIO.github.io/Mini_curso/tutorial.html`

A página inicial do site está disponível em [`index.html`](index.html) e contém um link direto para o tutorial.

## 1) Detect OS and Prerequisites

Check your OS and whether required tools are available:
- Package managers: `winget`, `brew`, or `apt`
- Core tools: `code`, `python`, `pip`

Quick checks:

```bash
code --version
python --version
pip --version
```

## 2) Install Visual Studio Code from CLI

### Windows (PowerShell)

```powershell
winget install --id Microsoft.VisualStudioCode -e --source winget
```

### macOS (Terminal)

```bash
brew install --cask visual-studio-code
```

### Ubuntu/Debian (Terminal)

```bash
sudo apt update
sudo apt install -y code
```

Validate:

```bash
code --version
```

## 3) Install Python and Validate pip

### Windows (PowerShell)

```powershell
winget install --id Python.Python.3.12 -e
```

### macOS (Terminal)

```bash
brew install python
```

### Ubuntu/Debian (Terminal)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

Validate:

```bash
python --version
python3 --version
py --version
pip --version
pip3 --version
```

## 4) Create Project Folder and Virtual Environment

### Windows (PowerShell)

```powershell
mkdir dev_workspace
cd dev_workspace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux (Bash/Zsh)

```bash
mkdir -p dev_workspace
cd dev_workspace
python3 -m venv .venv
source .venv/bin/activate
```

## 5) Install Core Python Developer Packages

With `.venv` activated:

```bash
python -m pip install --upgrade pip
pip install ipykernel pytest black flake8 pylint mypy
pip freeze > requirements.txt
```

## 6) Install VS Code Extensions with CLI

Install:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension ms-python.flake8
```

List installed extensions:

```bash
code --list-extensions
```

## 7) Create Workspace Settings for Python in VS Code

Create `.vscode/settings.json` with:
- Interpreter path pointing to `.venv`
- Black as default formatter
- Pytest enabled and configured to `tests`
- Environment auto-activation in terminal
- Notebook formatting preferences

## 8) Run Verification Script and Unit Tests

Create `hello_setup.py` and `tests/test_setup.py`, then run:

```bash
python hello_setup.py
pytest -q
```

Expected result:
- Python and core packages import correctly
- Pytest reports passing tests

## Troubleshooting

1. `python` not found:
   - Reopen terminal after install.
   - On Windows, reinstall Python and select "Add Python to PATH".
2. PowerShell cannot activate `.venv`:
   - Run:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

3. Wrong interpreter in VS Code:
   - Use Command Palette: `Python: Select Interpreter`, then choose `.venv`.
4. `code` command not found:
   - Reinstall VS Code with PATH integration enabled.

## Quick Daily Workflow

1. Open project folder in VS Code.
2. Activate `.venv` in terminal.
3. Run script or notebook work.
4. Format on save with Black.
5. Run `pytest -q` before commit.

## Pipeline completo de predição de atributos do solo

O script [`pipeline_predicao_solo.py`](pipeline_predicao_solo.py) cruza os atributos
e os espectros pela coluna `ID_Unique`, gera os comportamentos espectrais, aplica
SNV e primeira derivada dentro do pipeline de treino, escolhe o número de
componentes PLS por validação cruzada e salva métricas, predições e modelos.

Instale as dependências específicas:

```powershell
python -m pip install -r requirements-pipeline.txt
```

Execute VIS-NIR/SWIR (padrão):

```powershell
python pipeline_predicao_solo.py --spectra visnir --output-dir resultados_visnir
```

Também é possível executar NIR ou MIR:

```powershell
python pipeline_predicao_solo.py --spectra nir --output-dir resultados_nir
python pipeline_predicao_solo.py --spectra mir --output-dir resultados_mir
```

Cada pasta de saída contém `comportamentos_espectrais.png`, `metricas.csv`,
`predicoes_teste.csv`, `cv_<alvo>.csv`, `modelo_<alvo>.joblib`,
`wavelengths.csv` e `configuracao.json`. Os alvos podem ser restringidos com
`--targets Clay_gkg SOM_gkg pH_H2O`.
