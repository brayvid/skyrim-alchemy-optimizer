# Skyrim Alchemy Optimizer

<a href="https://colab.research.google.com/github/brayvid/skyrim-alchemy-optimizer/blob/main/skyrim_optimize_potions.ipynb" rel="Open in Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="" /></a>

<h4>Blake Rayvid - <a href=https://github.com/brayvid>https://github.com/brayvid</a></h4>

Make the most of the ingredients you have. Maximize total magnitude (essentially in-game value) with integer linear programming in scipy.



```python
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
```

## Read in ingredients and recipes
Uses local files "ingredients_have.csv" and "recipes_can_make.csv"

I made my CSVs using this helpful spreadsheet:<a href="https://docs.google.com/spreadsheets/d/1010C6ltqv7apuBoNYuFIFSBZER4YI03Y54kIsoKs5RI/edit?usp=sharing"> https://docs.google.com/spreadsheets/d/1010C6ltqv7apuBoNYuFIFSBZER4YI03Y54kIsoKs5RI/edit?usp=sharing </a>


```python
# Ingredients we have with quantity on hand
ingredients = pd.read_csv('ingredients_have.csv');ingredients
```





  <div id="df-1cc387aa-eeef-4267-a4f7-a9f6b7d3a607" class="colab-df-container">
    <div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ingredient</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Blisterwort</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blue Butterfly Wing</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blue Dartwing</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blue Mountain Flower</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bone Meal</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Butterfly Wing</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Canis Root</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Creep Cluster</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Deathbell</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dragons Tongue</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ectoplasm</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Elves Ear</td>
      <td>10</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Fire Salts</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Fly Amanita</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Frost Mirriam</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Garlic</td>
      <td>7</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Giant Lichen</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Glow Dust</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Hagraven Feathers</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Histcarp</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Honeycomb</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Ice Wraith Teeth</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Imp Stool</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Lavender</td>
      <td>17</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Luna Moth Wing</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Mora Tapinella</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Mudcrab Chitin</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nightshade</td>
      <td>7</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nirnroot</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Nordic Barnacle</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Orange Dartwing</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Purple Mountain Flower</td>
      <td>15</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Red Mountain Flower</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>River Betty</td>
      <td>2</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Rock Warbler Egg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Salt Pile</td>
      <td>14</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Scaly Pholiota</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Skeever Tail</td>
      <td>2</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Slaughterfish Scales</td>
      <td>3</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Snowberries</td>
      <td>6</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Spider Egg</td>
      <td>8</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Spriggan Sap</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Swamp Fungal Pod</td>
      <td>2</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Taproot</td>
      <td>2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Thistle Branch</td>
      <td>2</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Torchbug Thorax</td>
      <td>3</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Troll Fat</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tundra Cotton</td>
      <td>7</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Vampire Dust</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Void Salts</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>White Cap</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">








```python
# Potions list with magnitude and ingredient names (1,2 + optional 3rd)
recipes = pd.read_csv('recipes_can_make.csv')
recipes = recipes[recipes['Magnitude'] > 0]
recipes[['Magnitude','Ingredient 1','Ingredient 2','Ingredient 3','MyPotionID']].head(50)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Magnitude</th>
      <th>Ingredient 1</th>
      <th>Ingredient 2</th>
      <th>Ingredient 3</th>
      <th>MyPotionID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>159</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>3028</td>
    </tr>
    <tr>
      <th>1</th>
      <td>156</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Nightshade</td>
      <td>3037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>156</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>3045</td>
    </tr>
    <tr>
      <th>3</th>
      <td>156</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>3046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113</td>
      <td>Blisterwort</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>2130</td>
    </tr>
    <tr>
      <th>5</th>
      <td>113</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Rock Warbler Egg</td>
      <td>2680</td>
    </tr>
    <tr>
      <th>6</th>
      <td>112</td>
      <td>Frost Mirriam</td>
      <td>Histcarp</td>
      <td>Purple Mountain Flower</td>
      <td>10371</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Butterfly Wing</td>
      <td>2666</td>
    </tr>
    <tr>
      <th>8</th>
      <td>110</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Imp Stool</td>
      <td>2677</td>
    </tr>
    <tr>
      <th>9</th>
      <td>110</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Swamp Fungal Pod</td>
      <td>2684</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>River Betty</td>
      <td>11628</td>
    </tr>
    <tr>
      <th>11</th>
      <td>109</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>2210</td>
    </tr>
    <tr>
      <th>12</th>
      <td>109</td>
      <td>Blue Butterfly Wing</td>
      <td>Bone Meal</td>
      <td>Spriggan Sap</td>
      <td>2703</td>
    </tr>
    <tr>
      <th>13</th>
      <td>109</td>
      <td>Butterfly Wing</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>4738</td>
    </tr>
    <tr>
      <th>14</th>
      <td>109</td>
      <td>Creep Cluster</td>
      <td>Ectoplasm</td>
      <td>Histcarp</td>
      <td>6302</td>
    </tr>
    <tr>
      <th>15</th>
      <td>109</td>
      <td>Creep Cluster</td>
      <td>Histcarp</td>
      <td>Red Mountain Flower</td>
      <td>6550</td>
    </tr>
    <tr>
      <th>16</th>
      <td>109</td>
      <td>Creep Cluster</td>
      <td>River Betty</td>
      <td>Skeever Tail</td>
      <td>6725</td>
    </tr>
    <tr>
      <th>17</th>
      <td>109</td>
      <td>Nightshade</td>
      <td>River Betty</td>
      <td>Spriggan Sap</td>
      <td>14461</td>
    </tr>
    <tr>
      <th>18</th>
      <td>108</td>
      <td>Blisterwort</td>
      <td>Blue Butterfly Wing</td>
      <td>Spriggan Sap</td>
      <td>2154</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>2209</td>
    </tr>
    <tr>
      <th>20</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Bone Meal</td>
      <td>2665</td>
    </tr>
    <tr>
      <th>21</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Canis Root</td>
      <td>2667</td>
    </tr>
    <tr>
      <th>22</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Nirnroot</td>
      <td>2679</td>
    </tr>
    <tr>
      <th>23</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>2682</td>
    </tr>
    <tr>
      <th>24</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Bone Meal</td>
      <td>Glow Dust</td>
      <td>2693</td>
    </tr>
    <tr>
      <th>25</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Bone Meal</td>
      <td>Nightshade</td>
      <td>2700</td>
    </tr>
    <tr>
      <th>26</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Bone Meal</td>
      <td>Spider Egg</td>
      <td>2702</td>
    </tr>
    <tr>
      <th>27</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Glow Dust</td>
      <td>Hagraven Feathers</td>
      <td>2860</td>
    </tr>
    <tr>
      <th>28</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Hagraven Feathers</td>
      <td>Spider Egg</td>
      <td>2908</td>
    </tr>
    <tr>
      <th>29</th>
      <td>108</td>
      <td>Blue Butterfly Wing</td>
      <td>Lavender</td>
      <td>Spider Egg</td>
      <td>2969</td>
    </tr>
    <tr>
      <th>30</th>
      <td>108</td>
      <td>Blue Dartwing</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>3203</td>
    </tr>
    <tr>
      <th>31</th>
      <td>108</td>
      <td>Blue Mountain Flower</td>
      <td>Bone Meal</td>
      <td>Spider Egg</td>
      <td>3416</td>
    </tr>
    <tr>
      <th>32</th>
      <td>108</td>
      <td>Blue Mountain Flower</td>
      <td>Butterfly Wing</td>
      <td>Glow Dust</td>
      <td>3429</td>
    </tr>
    <tr>
      <th>33</th>
      <td>108</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Hagraven Feathers</td>
      <td>3628</td>
    </tr>
    <tr>
      <th>34</th>
      <td>108</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Swamp Fungal Pod</td>
      <td>3643</td>
    </tr>
    <tr>
      <th>35</th>
      <td>108</td>
      <td>Blue Mountain Flower</td>
      <td>Rock Warbler Egg</td>
      <td>Spider Egg</td>
      <td>3780</td>
    </tr>
    <tr>
      <th>36</th>
      <td>108</td>
      <td>Glow Dust</td>
      <td>Hagraven Feathers</td>
      <td>Nightshade</td>
      <td>11513</td>
    </tr>
    <tr>
      <th>37</th>
      <td>108</td>
      <td>Glow Dust</td>
      <td>Luna Moth Wing</td>
      <td>Nightshade</td>
      <td>11598</td>
    </tr>
    <tr>
      <th>38</th>
      <td>108</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>Nordic Barnacle</td>
      <td>11624</td>
    </tr>
    <tr>
      <th>39</th>
      <td>108</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>Snowberries</td>
      <td>11631</td>
    </tr>
    <tr>
      <th>40</th>
      <td>108</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>Swamp Fungal Pod</td>
      <td>11632</td>
    </tr>
    <tr>
      <th>41</th>
      <td>107</td>
      <td>Blisterwort</td>
      <td>Spider Egg</td>
      <td>Spriggan Sap</td>
      <td>2648</td>
    </tr>
    <tr>
      <th>42</th>
      <td>107</td>
      <td>Blue Butterfly Wing</td>
      <td>Canis Root</td>
      <td>Spider Egg</td>
      <td>2724</td>
    </tr>
    <tr>
      <th>43</th>
      <td>107</td>
      <td>Blue Mountain Flower</td>
      <td>Imp Stool</td>
      <td>Nightshade</td>
      <td>3716</td>
    </tr>
    <tr>
      <th>44</th>
      <td>107</td>
      <td>Canis Root</td>
      <td>Spider Egg</td>
      <td>Spriggan Sap</td>
      <td>5163</td>
    </tr>
    <tr>
      <th>45</th>
      <td>107</td>
      <td>Creep Cluster</td>
      <td>Ectoplasm</td>
      <td>Skeever Tail</td>
      <td>6318</td>
    </tr>
    <tr>
      <th>46</th>
      <td>107</td>
      <td>Creep Cluster</td>
      <td>Frost Mirriam</td>
      <td>Purple Mountain Flower</td>
      <td>6392</td>
    </tr>
    <tr>
      <th>47</th>
      <td>107</td>
      <td>Creep Cluster</td>
      <td>Frost Mirriam</td>
      <td>Red Mountain Flower</td>
      <td>6393</td>
    </tr>
    <tr>
      <th>48</th>
      <td>107</td>
      <td>Creep Cluster</td>
      <td>Frost Mirriam</td>
      <td>Taproot</td>
      <td>6402</td>
    </tr>
    <tr>
      <th>49</th>
      <td>107</td>
      <td>Creep Cluster</td>
      <td>Frost Mirriam</td>
      <td>White Cap</td>
      <td>6406</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">



## Create recipe matrix A in Ax <= b
One row for each ingredient, one column for each potion. "1" indicates the ingredient is used in the potion.




```python
# Boolean matrix A says what ingredients are in what recipes
A = pd.DataFrame(0, index=range(len(ingredients)),columns=range(len(recipes)))
for i in range(len(recipes)):
  if ingredients.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 1"]).idxmax()]["Quantity"] > 0:
    A.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 1"]).idxmax(), i] = 1
  if ingredients.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 2"]).idxmax()]["Quantity"] > 0:
    A.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 2"]).idxmax(), i] = 1
  if not pd.isnull(recipes.loc[i, "Ingredient 3"]):
    if ingredients.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 3"]).idxmax()]["Quantity"] > 0:
      A.iloc[ingredients["Ingredient"].str.find(recipes.loc[i, "Ingredient 3"]).idxmax(), i] = 1
A
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>2375</th>
      <th>2376</th>
      <th>2377</th>
      <th>2378</th>
      <th>2379</th>
      <th>2380</th>
      <th>2381</th>
      <th>2382</th>
      <th>2383</th>
      <th>2384</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>51 rows × 2385 columns</p>
</div>
    <div class="colab-df-buttons">





## Set up optimization variables
find x to minimize f.x with Ax <= b, x >= lb

f = -1 * magnitude, b = qty of each ingredient on hand


```python
# Objective function f.x to minimize
f = np.array(-1 * recipes['Magnitude'],dtype=int); # f = -1*value so that minimizing f.x maximizes total value
```


```python
# Bounds
b_max = np.array(ingredients['Quantity'],dtype=int) # Cannot use more than we have on hand
x_lb = np.zeros(shape=len(recipes)) # Cannot use less than 0
```


```python
# milp parameters
bounds = Bounds(lb=x_lb)
constraint = LinearConstraint(A, ub=b_max)
integrality = np.ones(shape=len(recipes),dtype=int) # All x should be integers
```

# Perform optimization with scipy.optimize.milp


```python
# Perform optimization
res = milp(c=f, integrality=integrality, bounds=bounds, constraints=constraint)
```

## Display recommended potions to make


```python
# Display the potions we should make to maximize magnitude where the last column is quantity to make
total_magnitude = int(-res.fun)
num_potions = int(sum(res.x))
indices_to_make = np.nonzero(res.x > 0)
to_make_df = recipes.iloc[indices_to_make].copy()
to_make_df.loc[:,'QtyToMake'] = res.x[indices_to_make].astype(int)
to_make_df[['Magnitude','Type','Ingredient 1','Ingredient 2','Ingredient 3','MyPotionID','QtyToMake']].head(50)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Magnitude</th>
      <th>Type</th>
      <th>Ingredient 1</th>
      <th>Ingredient 2</th>
      <th>Ingredient 3</th>
      <th>MyPotionID</th>
      <th>QtyToMake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>112</td>
      <td>Mixed</td>
      <td>Frost Mirriam</td>
      <td>Histcarp</td>
      <td>Purple Mountain Flower</td>
      <td>10371</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Butterfly Wing</td>
      <td>2666</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>2210</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>2209</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Bone Meal</td>
      <td>Spider Egg</td>
      <td>3416</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Hagraven Feathers</td>
      <td>3628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Swamp Fungal Pod</td>
      <td>3643</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Rock Warbler Egg</td>
      <td>Spider Egg</td>
      <td>3780</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>107</td>
      <td>Mixed</td>
      <td>Creep Cluster</td>
      <td>Ectoplasm</td>
      <td>Skeever Tail</td>
      <td>6318</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>107</td>
      <td>Mixed</td>
      <td>Frost Mirriam</td>
      <td>Purple Mountain Flower</td>
      <td>Skeever Tail</td>
      <td>10519</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>105</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Lavender</td>
      <td>Nightshade</td>
      <td>3749</td>
      <td>7</td>
    </tr>
    <tr>
      <th>104</th>
      <td>105</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Lavender</td>
      <td>Spider Egg</td>
      <td>3755</td>
      <td>2</td>
    </tr>
    <tr>
      <th>303</th>
      <td>59</td>
      <td>Potion</td>
      <td>Blue Dartwing</td>
      <td>Swamp Fungal Pod</td>
      <td>NaN</td>
      <td>3388</td>
      <td>1</td>
    </tr>
    <tr>
      <th>334</th>
      <td>57</td>
      <td>Mixed</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>Taproot</td>
      <td>8001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>361</th>
      <td>55</td>
      <td>Poison</td>
      <td>River Betty</td>
      <td>Salt Pile</td>
      <td>Troll Fat</td>
      <td>15006</td>
      <td>2</td>
    </tr>
    <tr>
      <th>367</th>
      <td>53</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Nirnroot</td>
      <td>Salt Pile</td>
      <td>7937</td>
      <td>2</td>
    </tr>
    <tr>
      <th>370</th>
      <td>53</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>Troll Fat</td>
      <td>8004</td>
      <td>1</td>
    </tr>
    <tr>
      <th>379</th>
      <td>50</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>NaN</td>
      <td>8006</td>
      <td>2</td>
    </tr>
    <tr>
      <th>384</th>
      <td>17</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Fire Salts</td>
      <td>Salt Pile</td>
      <td>8930</td>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>16</td>
      <td>Potion</td>
      <td>Dragons Tongue</td>
      <td>Fly Amanita</td>
      <td>Scaly Pholiota</td>
      <td>8094</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Taproot</td>
      <td>Vampire Dust</td>
      <td>11060</td>
      <td>1</td>
    </tr>
    <tr>
      <th>400</th>
      <td>14</td>
      <td>Mixed</td>
      <td>Ectoplasm</td>
      <td>Giant Lichen</td>
      <td>Void Salts</td>
      <td>8576</td>
      <td>1</td>
    </tr>
    <tr>
      <th>443</th>
      <td>12</td>
      <td>Mixed</td>
      <td>Canis Root</td>
      <td>Imp Stool</td>
      <td>Rock Warbler Egg</td>
      <td>5088</td>
      <td>2</td>
    </tr>
    <tr>
      <th>461</th>
      <td>12</td>
      <td>Mixed</td>
      <td>Luna Moth Wing</td>
      <td>Nordic Barnacle</td>
      <td>Orange Dartwing</td>
      <td>14094</td>
      <td>1</td>
    </tr>
    <tr>
      <th>465</th>
      <td>12</td>
      <td>Potion</td>
      <td>Dragons Tongue</td>
      <td>Elves Ear</td>
      <td>Mora Tapinella</td>
      <td>8058</td>
      <td>2</td>
    </tr>
    <tr>
      <th>468</th>
      <td>12</td>
      <td>Potion</td>
      <td>Honeycomb</td>
      <td>Purple Mountain Flower</td>
      <td>Slaughterfish Scales</td>
      <td>12905</td>
      <td>1</td>
    </tr>
    <tr>
      <th>469</th>
      <td>12</td>
      <td>Potion</td>
      <td>Mudcrab Chitin</td>
      <td>Purple Mountain Flower</td>
      <td>Thistle Branch</td>
      <td>14343</td>
      <td>2</td>
    </tr>
    <tr>
      <th>470</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Ectoplasm</td>
      <td>Red Mountain Flower</td>
      <td>NaN</td>
      <td>8873</td>
      <td>1</td>
    </tr>
    <tr>
      <th>493</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Dragons Tongue</td>
      <td>Elves Ear</td>
      <td>White Cap</td>
      <td>8067</td>
      <td>2</td>
    </tr>
    <tr>
      <th>516</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Snowberries</td>
      <td>White Cap</td>
      <td>9130</td>
      <td>2</td>
    </tr>
    <tr>
      <th>582</th>
      <td>10</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Ice Wraith Teeth</td>
      <td>White Cap</td>
      <td>9037</td>
      <td>2</td>
    </tr>
    <tr>
      <th>658</th>
      <td>9</td>
      <td>Mixed</td>
      <td>Bone Meal</td>
      <td>Lavender</td>
      <td>Nirnroot</td>
      <td>4095</td>
      <td>1</td>
    </tr>
    <tr>
      <th>704</th>
      <td>9</td>
      <td>Mixed</td>
      <td>Purple Mountain Flower</td>
      <td>Snowberries</td>
      <td>Torchbug Thorax</td>
      <td>14933</td>
      <td>3</td>
    </tr>
    <tr>
      <th>760</th>
      <td>9</td>
      <td>Potion</td>
      <td>Ectoplasm</td>
      <td>Elves Ear</td>
      <td>Tundra Cotton</td>
      <td>8459</td>
      <td>1</td>
    </tr>
    <tr>
      <th>765</th>
      <td>9</td>
      <td>Potion</td>
      <td>Ectoplasm</td>
      <td>Giant Lichen</td>
      <td>Tundra Cotton</td>
      <td>8575</td>
      <td>1</td>
    </tr>
    <tr>
      <th>804</th>
      <td>9</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Lavender</td>
      <td>Luna Moth Wing</td>
      <td>10953</td>
      <td>1</td>
    </tr>
    <tr>
      <th>806</th>
      <td>9</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Lavender</td>
      <td>Salt Pile</td>
      <td>10961</td>
      <td>5</td>
    </tr>
    <tr>
      <th>855</th>
      <td>9</td>
      <td>Potion</td>
      <td>Honeycomb</td>
      <td>Purple Mountain Flower</td>
      <td>Tundra Cotton</td>
      <td>12911</td>
      <td>3</td>
    </tr>
    <tr>
      <th>879</th>
      <td>8</td>
      <td>Mixed</td>
      <td>Luna Moth Wing</td>
      <td>Nordic Barnacle</td>
      <td>NaN</td>
      <td>14098</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>8</td>
      <td>Mixed</td>
      <td>Hagraven Feathers</td>
      <td>Lavender</td>
      <td>Luna Moth Wing</td>
      <td>12101</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1144</th>
      <td>8</td>
      <td>Potion</td>
      <td>Orange Dartwing</td>
      <td>Purple Mountain Flower</td>
      <td>Snowberries</td>
      <td>14612</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>7</td>
      <td>Potion</td>
      <td>Purple Mountain Flower</td>
      <td>Slaughterfish Scales</td>
      <td>Tundra Cotton</td>
      <td>14922</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

```python
print(f"To maximize magnitude and therefore value, create {num_potions} potions of the {len(to_make_df)} unique types listed above for a total magnitude of {total_magnitude}.")
```

    To maximize magnitude and therefore value, create 76 potions of the 42 unique types listed above for a total magnitude of 3905.

