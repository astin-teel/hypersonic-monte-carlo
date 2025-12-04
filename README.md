# hypersonic-monte-carlo

20251203 | Teel, Astin
# Hypersonic Monte Carlo Dispersion Analysis
6-DOF-style analystic boost-glide simulation with realistic Monte Carlo dispersions (velocity, flight path angle, drag, mass, wind).
Generates 5,000 case instantaneous impact point (IIP) footprint and 99% containment ellipse per RCC 321 standards.

## Features
- Realistic hypersonic dispersions (3 sigma values from open literature)
- 99% probabililty ellipse using chi-sqaure method
- <90s runtime on laptop
- Easily extensible to full numerical 6DOF

  ## Quick Start
  '''bash
  git clone https://github.com/astinteel/hypersonic-monte-carlo.git
  cd hypersonic-monte-carlo
  pip install -r requirements.txt
  python main.py
