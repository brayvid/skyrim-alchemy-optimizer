# Skyrim Alchemy Optimizer

<a href="https://colab.research.google.com/github/brayvid/skyrim-alchemy-optimizer/blob/main/skyrim_optimize_potions.ipynb" rel="Open in Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="" /></a>

A Colab notebook which can be used to maximize alchemy profitability using the ingredients you have on hand in The Elder Scrolls V: Skyrim.

It uses the integer linear programming function [scipy.optimize.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), and requires a csv file of the ingredients and quantities you have, and a csv file of all possible potions you could make. I used [this helpful spreadsheet](https://docs.google.com/spreadsheets/d/1010C6ltqv7apuBoNYuFIFSBZER4YI03Y54kIsoKs5RI/edit?usp=sharing) to create my csvs, which are available here as examples.
