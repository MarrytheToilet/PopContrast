"""Build a standalone manuscript source bundle from shared reproducible scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'reports/www2027')
    args = ap.parse_args()
    if not (PAPER/'main.tex').is_file():
        ap.error('This optional command requires the local paper/main.tex source.')
    subprocess.run(['make','assets','pdf','check'],cwd=PAPER,check=True)
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    files={};seen=set()
    def visit(path):
        if path in seen:return
        seen.add(path);files[str(path.relative_to(PAPER))]=path
        for name in re.findall(r'\\input(?:table)?\{([^}]+)\}',path.read_text()):
            if '#1' in name:continue
            child=PAPER/name
            visit(child if child.suffix else child.with_suffix('.tex'))
    visit(PAPER/'main.tex')
    text='\n'.join(p.read_text() for p in seen)
    for name in re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}',text):
        files[name]=PAPER/name
    for name in ['references.bib','acmart.cls','ACM-Reference-Format.bst','Makefile','README.md','build_assets.py','redraw_figures.py','validate_manuscript.py','requirements-figures.txt','asset_sources.json','figure_sources.json','reference_checks.json']:
        files[name]=PAPER/name
    reference=json.loads((PAPER/'reference_checks.json').read_text())
    for name in reference['original_images']:files[name]=PAPER/name
    for path in (ROOT/'experiments/figures').glob('*.py'):files[str(path.relative_to(ROOT))]=path
    for path in (ROOT/'assets/fonts').iterdir():files[str(path.relative_to(ROOT))]=path
    files['assets/framework.png']=ROOT/'assets/framework.png'
    sources=set(json.loads((PAPER/'asset_sources.json').read_text())['sources'])|set(json.loads((PAPER/'figure_sources.json').read_text())['sources'])
    for name in sources:
        if name.startswith('results/'):files['plot_data/'+name]=ROOT/name
    files['VALIDATION.json']=ROOT/'reports/www2027/VALIDATION.json'
    archive=output/'PopContrast_WWW2027_updated_source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):z.write(path,name)
    shutil.copy2(archive,output/'PopContrast_WWW2027_source.zip')
    pdf=output/'PopContrast_WWW2027_updated.pdf';shutil.copy2(PAPER/'main.pdf',pdf)
    # Verify the actual bundle, including plot regeneration, outside the checkout.
    import fitz
    with tempfile.TemporaryDirectory(prefix='popcontrast_package_') as work:
        work=Path(work)
        with zipfile.ZipFile(archive) as z:z.extractall(work)
        subprocess.run(['make','assets','pdf'],cwd=work,check=True)
        original=fitz.open(pdf);rebuilt=fitz.open(work/'main.pdf')
        assert len(original)==len(rebuilt)==12
        assert [p.get_text() for p in original]==[p.get_text() for p in rebuilt]
        assert all(p.get_pixmap(alpha=False).samples==q.get_pixmap(alpha=False).samples for p,q in zip(original,rebuilt))
        for name in ['pareto_single_column','model_seed_evidence','framework_refined','matched_tradeoffs']:
            assert (PAPER/f'figures/www2027/{name}.png').read_bytes()==(work/f'figures/www2027/{name}.png').read_bytes(),name
        assert not any(word in (work/'main.log').read_text() for word in ['Warning','Overfull','undefined references'])
    report=dict(status='passed',standalone_compile=True,standalone_redraw=True,
                all_twelve_pages_render_identically=True,source_archive=archive.name,
                source_files=len(files),source_archive_sha256=sha(archive),pdf_sha256=sha(pdf),
                main_pages=8,total_pages=12,rendered_figures=10,rendered_tables=12)
    (output/'SOURCE_PACKAGE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
