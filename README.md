<p align="center">
  <h1 align="center"> <ins>DaD:</ins> Distilled Reinforcement Learning for Diverse Keypoint Detection</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="TBD" align="center">Paper (Coming Soon...)</a>
  </p></h2>
  <div align="center">DaD's a pretty good keypoint detector, probably the best.</div>
</p>
<p align="center">
</p>

## Run
```python
import dad
detector = dad.load_DaD()
keypoints = detector.detect_from_path(
  "assets/0015_A.jpg", 
  num_keypoints = 512) # 1 x 512 x 2
```

## Install
Get uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Either clone and install editable
```bash
git clone git@github.com:Parskatt/dad.git
uv pip install -e .
```
**or** install 
```bash
uv pip install dad@git+https://github.com/Parskatt/dad.git
```



## Licenses
Most of the code is MIT licensed.
Third party detectors in [dad/detectors/third_party](dad/detectors/third_party) have their own licenses. If you use them, please refer to their respective licenses in [here](licenses).

## BibTeX

```txt
Coming soon...
```
