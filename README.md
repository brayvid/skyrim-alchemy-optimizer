# Skyrim Alchemy Optimizer

<a href="https://colab.research.google.com/github/brayvid/skyrim-alchemy-optimizer/blob/main/skyrim_optimize_potions.ipynb" rel="Open in Colab"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="" /></a>

A Colab notebook which can be used to maximize alchemy profitability using the ingredients you have on hand in The Elder Scrolls V: Skyrim.

It uses the integer linear programming function [scipy.optimize.milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), and requires a csv file of the ingredients and quantities you have, and a csv file of all possible potions you could make. I used [this helpful spreadsheet](https://docs.google.com/spreadsheets/d/1010C6ltqv7apuBoNYuFIFSBZER4YI03Y54kIsoKs5RI/edit?usp=sharing) to create my csvs, which are available here as examples.


<h1>Skyrim Alchemy Optimization</h1>
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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1cc387aa-eeef-4267-a4f7-a9f6b7d3a607')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1cc387aa-eeef-4267-a4f7-a9f6b7d3a607 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1cc387aa-eeef-4267-a4f7-a9f6b7d3a607');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-55d2926a-6e27-43fc-9f36-daf955eb360f">
  <button class="colab-df-quickchart" onclick="quickchart('df-55d2926a-6e27-43fc-9f36-daf955eb360f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-55d2926a-6e27-43fc-9f36-daf955eb360f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_dd37a224-ff67-4ea4-ac85-ee6110c52322">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('ingredients')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_dd37a224-ff67-4ea4-ac85-ee6110c52322 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('ingredients');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Potions list with magnitude and ingredient names (1,2 + optional 3rd)
recipes = pd.read_csv('recipes_can_make.csv')
recipes = recipes[recipes['Magnitude'] > 0];
recipes.head(20)
```





  <div id="df-e2e5df58-924b-467f-9cbc-fd90d170e289" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Magnitude</th>
      <th>Type</th>
      <th>Ingredient 1</th>
      <th>Ingredient 2</th>
      <th>Ingredient 3</th>
      <th>Effects</th>
      <th>Effect 1</th>
      <th>Effect 2</th>
      <th>Effect 3</th>
      <th>Effect 4</th>
      <th>Effect 5</th>
      <th>MyPotionID</th>
      <th>Can Make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>159</td>
      <td>Mixed</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>Resist Shock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3028</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>156</td>
      <td>Mixed</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Nightshade</td>
      <td>2</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3037</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>156</td>
      <td>Mixed</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>2</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3045</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>156</td>
      <td>Mixed</td>
      <td>Blue Dartwing</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>2</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3046</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>4</td>
      <td>Damage Stamina</td>
      <td>Restore Health</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>2130</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>113</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Rock Warbler Egg</td>
      <td>4</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Restore Health</td>
      <td>Damage Stamina</td>
      <td>NaN</td>
      <td>2680</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>112</td>
      <td>Mixed</td>
      <td>Frost Mirriam</td>
      <td>Histcarp</td>
      <td>Purple Mountain Flower</td>
      <td>4</td>
      <td>Damage Stamina Regen</td>
      <td>Restore Stamina</td>
      <td>Fortify Sneak</td>
      <td>Resist Frost</td>
      <td>NaN</td>
      <td>10371</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Butterfly Wing</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2666</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Imp Stool</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2677</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Swamp Fungal Pod</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2684</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>River Betty</td>
      <td>3</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Destruction</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11628</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Smithing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2210</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Bone Meal</td>
      <td>Spriggan Sap</td>
      <td>4</td>
      <td>Damage Stamina</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Enchanting</td>
      <td>NaN</td>
      <td>2703</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Butterfly Wing</td>
      <td>Glow Dust</td>
      <td>Nightshade</td>
      <td>4</td>
      <td>Damage Magicka</td>
      <td>Damage Magicka Regen</td>
      <td>Lingering Damage Stamina</td>
      <td>Fortify Destruction</td>
      <td>NaN</td>
      <td>4738</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Creep Cluster</td>
      <td>Ectoplasm</td>
      <td>Histcarp</td>
      <td>3</td>
      <td>Restore Magicka</td>
      <td>Fortify Magicka</td>
      <td>Damage Stamina Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6302</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Creep Cluster</td>
      <td>Histcarp</td>
      <td>Red Mountain Flower</td>
      <td>3</td>
      <td>Damage Stamina Regen</td>
      <td>Restore Magicka</td>
      <td>Fortify Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6550</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Creep Cluster</td>
      <td>River Betty</td>
      <td>Skeever Tail</td>
      <td>3</td>
      <td>Fortify Carry Weight</td>
      <td>Damage Stamina Regen</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6725</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Nightshade</td>
      <td>River Betty</td>
      <td>Spriggan Sap</td>
      <td>3</td>
      <td>Damage Health</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Alteration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14461</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Butterfly Wing</td>
      <td>Spriggan Sap</td>
      <td>4</td>
      <td>Damage Stamina</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Enchanting</td>
      <td>Fortify Smithing</td>
      <td>NaN</td>
      <td>2154</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Stamina</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2209</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e2e5df58-924b-467f-9cbc-fd90d170e289')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e2e5df58-924b-467f-9cbc-fd90d170e289 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e2e5df58-924b-467f-9cbc-fd90d170e289');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-bdc26e3e-1765-415e-88c6-17407abe3f75">
  <button class="colab-df-quickchart" onclick="quickchart('df-bdc26e3e-1765-415e-88c6-17407abe3f75')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-bdc26e3e-1765-415e-88c6-17407abe3f75 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




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
A.head(20)
```





  <div id="df-4bdf036d-56a4-4c44-b112-b87fb7cf1cb3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
  </tbody>
</table>
<p>20 rows × 2385 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4bdf036d-56a4-4c44-b112-b87fb7cf1cb3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4bdf036d-56a4-4c44-b112-b87fb7cf1cb3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4bdf036d-56a4-4c44-b112-b87fb7cf1cb3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f2624c12-234a-4255-93e0-954d9dd0e09d">
  <button class="colab-df-quickchart" onclick="quickchart('df-f2624c12-234a-4255-93e0-954d9dd0e09d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f2624c12-234a-4255-93e0-954d9dd0e09d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




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
to_make_df.head(len(to_make_df))
```





  <div id="df-71193b63-0f32-4744-9859-bc6cce0fd0d2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Magnitude</th>
      <th>Type</th>
      <th>Ingredient 1</th>
      <th>Ingredient 2</th>
      <th>Ingredient 3</th>
      <th>Effects</th>
      <th>Effect 1</th>
      <th>Effect 2</th>
      <th>Effect 3</th>
      <th>Effect 4</th>
      <th>Effect 5</th>
      <th>MyPotionID</th>
      <th>Can Make</th>
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
      <td>4</td>
      <td>Damage Stamina Regen</td>
      <td>Restore Stamina</td>
      <td>Fortify Sneak</td>
      <td>Resist Frost</td>
      <td>NaN</td>
      <td>10371</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110</td>
      <td>Mixed</td>
      <td>Blue Butterfly Wing</td>
      <td>Blue Mountain Flower</td>
      <td>Butterfly Wing</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2666</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>109</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spriggan Sap</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Magicka Regen</td>
      <td>Fortify Smithing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2210</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blisterwort</td>
      <td>Blue Mountain Flower</td>
      <td>Spider Egg</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Stamina</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2209</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Bone Meal</td>
      <td>Spider Egg</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Stamina</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3416</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Hagraven Feathers</td>
      <td>3</td>
      <td>Damage Magicka Regen</td>
      <td>Damage Magicka</td>
      <td>Fortify Conjuration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3628</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Glow Dust</td>
      <td>Swamp Fungal Pod</td>
      <td>3</td>
      <td>Damage Magicka Regen</td>
      <td>Resist Shock</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3643</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>108</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Rock Warbler Egg</td>
      <td>Spider Egg</td>
      <td>3</td>
      <td>Restore Health</td>
      <td>Damage Stamina</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3780</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>107</td>
      <td>Mixed</td>
      <td>Creep Cluster</td>
      <td>Ectoplasm</td>
      <td>Skeever Tail</td>
      <td>3</td>
      <td>Restore Magicka</td>
      <td>Damage Stamina Regen</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6318</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>57</th>
      <td>107</td>
      <td>Mixed</td>
      <td>Frost Mirriam</td>
      <td>Purple Mountain Flower</td>
      <td>Skeever Tail</td>
      <td>3</td>
      <td>Resist Frost</td>
      <td>Fortify Sneak</td>
      <td>Damage Stamina Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10519</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>105</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Lavender</td>
      <td>Nightshade</td>
      <td>2</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3749</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>104</th>
      <td>105</td>
      <td>Mixed</td>
      <td>Blue Mountain Flower</td>
      <td>Lavender</td>
      <td>Spider Egg</td>
      <td>2</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka Regen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3755</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>303</th>
      <td>59</td>
      <td>Potion</td>
      <td>Blue Dartwing</td>
      <td>Swamp Fungal Pod</td>
      <td>NaN</td>
      <td>2</td>
      <td>Resist Shock</td>
      <td>Restore Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3388</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>334</th>
      <td>57</td>
      <td>Mixed</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>Taproot</td>
      <td>3</td>
      <td>Slow</td>
      <td>Weakness to Magic</td>
      <td>Regenerate Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8001</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>361</th>
      <td>55</td>
      <td>Poison</td>
      <td>River Betty</td>
      <td>Salt Pile</td>
      <td>Troll Fat</td>
      <td>2</td>
      <td>Slow</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15006</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>367</th>
      <td>53</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Nirnroot</td>
      <td>Salt Pile</td>
      <td>2</td>
      <td>Damage Health</td>
      <td>Slow</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7937</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>370</th>
      <td>53</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>Troll Fat</td>
      <td>2</td>
      <td>Slow</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8004</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>379</th>
      <td>50</td>
      <td>Poison</td>
      <td>Deathbell</td>
      <td>Salt Pile</td>
      <td>NaN</td>
      <td>1</td>
      <td>Slow</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8006</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>384</th>
      <td>17</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Fire Salts</td>
      <td>Salt Pile</td>
      <td>4</td>
      <td>Restore Magicka</td>
      <td>Weakness to Frost</td>
      <td>Resist Fire</td>
      <td>Regenerate Magicka</td>
      <td>NaN</td>
      <td>8930</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>16</td>
      <td>Potion</td>
      <td>Dragons Tongue</td>
      <td>Fly Amanita</td>
      <td>Scaly Pholiota</td>
      <td>4</td>
      <td>Resist Fire</td>
      <td>Fortify Two-handed</td>
      <td>Fortify Illusion</td>
      <td>Regenerate Stamina</td>
      <td>NaN</td>
      <td>8094</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>15</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Taproot</td>
      <td>Vampire Dust</td>
      <td>3</td>
      <td>Regenerate Magicka</td>
      <td>Restore Magicka</td>
      <td>Regenerate Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11060</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>400</th>
      <td>14</td>
      <td>Mixed</td>
      <td>Ectoplasm</td>
      <td>Giant Lichen</td>
      <td>Void Salts</td>
      <td>4</td>
      <td>Restore Magicka</td>
      <td>Weakness to Shock</td>
      <td>Damage Health</td>
      <td>Fortify Magicka</td>
      <td>NaN</td>
      <td>8576</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>443</th>
      <td>12</td>
      <td>Mixed</td>
      <td>Canis Root</td>
      <td>Imp Stool</td>
      <td>Rock Warbler Egg</td>
      <td>4</td>
      <td>Paralysis</td>
      <td>Restore Health</td>
      <td>Fortify One-Handed</td>
      <td>Damage Stamina</td>
      <td>NaN</td>
      <td>5088</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>461</th>
      <td>12</td>
      <td>Mixed</td>
      <td>Luna Moth Wing</td>
      <td>Nordic Barnacle</td>
      <td>Orange Dartwing</td>
      <td>3</td>
      <td>Damage Magicka</td>
      <td>Regenerate Health</td>
      <td>Fortify Pickpocket</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14094</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>465</th>
      <td>12</td>
      <td>Potion</td>
      <td>Dragons Tongue</td>
      <td>Elves Ear</td>
      <td>Mora Tapinella</td>
      <td>3</td>
      <td>Resist Fire</td>
      <td>Restore Magicka</td>
      <td>Fortify Illusion</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8058</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>468</th>
      <td>12</td>
      <td>Potion</td>
      <td>Honeycomb</td>
      <td>Purple Mountain Flower</td>
      <td>Slaughterfish Scales</td>
      <td>3</td>
      <td>Restore Stamina</td>
      <td>Resist Frost</td>
      <td>Fortify Block</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12905</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>469</th>
      <td>12</td>
      <td>Potion</td>
      <td>Mudcrab Chitin</td>
      <td>Purple Mountain Flower</td>
      <td>Thistle Branch</td>
      <td>3</td>
      <td>Restore Stamina</td>
      <td>Resist Frost</td>
      <td>Resist Poison</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14343</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>470</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Ectoplasm</td>
      <td>Red Mountain Flower</td>
      <td>NaN</td>
      <td>3</td>
      <td>Restore Magicka</td>
      <td>Fortify Magicka</td>
      <td>Damage Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8873</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>493</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Dragons Tongue</td>
      <td>Elves Ear</td>
      <td>White Cap</td>
      <td>3</td>
      <td>Resist Fire</td>
      <td>Weakness to Frost</td>
      <td>Restore Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8067</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>516</th>
      <td>11</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Snowberries</td>
      <td>White Cap</td>
      <td>3</td>
      <td>Resist Fire</td>
      <td>Weakness to Frost</td>
      <td>Restore Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9130</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>582</th>
      <td>10</td>
      <td>Mixed</td>
      <td>Elves Ear</td>
      <td>Ice Wraith Teeth</td>
      <td>White Cap</td>
      <td>3</td>
      <td>Weakness to Frost</td>
      <td>Fortify Heavy Armor</td>
      <td>Restore Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9037</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>658</th>
      <td>9</td>
      <td>Mixed</td>
      <td>Bone Meal</td>
      <td>Lavender</td>
      <td>Nirnroot</td>
      <td>3</td>
      <td>Fortify Conjuration</td>
      <td>Damage Stamina</td>
      <td>Resist Magic</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4095</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>704</th>
      <td>9</td>
      <td>Mixed</td>
      <td>Purple Mountain Flower</td>
      <td>Snowberries</td>
      <td>Torchbug Thorax</td>
      <td>3</td>
      <td>Resist Frost</td>
      <td>Restore Stamina</td>
      <td>Lingering Damage Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14933</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>760</th>
      <td>9</td>
      <td>Potion</td>
      <td>Ectoplasm</td>
      <td>Elves Ear</td>
      <td>Tundra Cotton</td>
      <td>2</td>
      <td>Restore Magicka</td>
      <td>Fortify Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8459</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>765</th>
      <td>9</td>
      <td>Potion</td>
      <td>Ectoplasm</td>
      <td>Giant Lichen</td>
      <td>Tundra Cotton</td>
      <td>2</td>
      <td>Restore Magicka</td>
      <td>Fortify Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8575</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>804</th>
      <td>9</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Lavender</td>
      <td>Luna Moth Wing</td>
      <td>2</td>
      <td>Fortify Stamina</td>
      <td>Regenerate Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10953</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>806</th>
      <td>9</td>
      <td>Potion</td>
      <td>Garlic</td>
      <td>Lavender</td>
      <td>Salt Pile</td>
      <td>2</td>
      <td>Fortify Stamina</td>
      <td>Regenerate Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10961</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>855</th>
      <td>9</td>
      <td>Potion</td>
      <td>Honeycomb</td>
      <td>Purple Mountain Flower</td>
      <td>Tundra Cotton</td>
      <td>2</td>
      <td>Restore Stamina</td>
      <td>Fortify Block</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12911</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>879</th>
      <td>8</td>
      <td>Mixed</td>
      <td>Luna Moth Wing</td>
      <td>Nordic Barnacle</td>
      <td>NaN</td>
      <td>2</td>
      <td>Damage Magicka</td>
      <td>Regenerate Health</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14098</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>8</td>
      <td>Mixed</td>
      <td>Hagraven Feathers</td>
      <td>Lavender</td>
      <td>Luna Moth Wing</td>
      <td>2</td>
      <td>Fortify Conjuration</td>
      <td>Damage Magicka</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12101</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1144</th>
      <td>8</td>
      <td>Potion</td>
      <td>Orange Dartwing</td>
      <td>Purple Mountain Flower</td>
      <td>Snowberries</td>
      <td>2</td>
      <td>Restore Stamina</td>
      <td>Resist Frost</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14612</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>7</td>
      <td>Potion</td>
      <td>Purple Mountain Flower</td>
      <td>Slaughterfish Scales</td>
      <td>Tundra Cotton</td>
      <td>2</td>
      <td>Resist Frost</td>
      <td>Fortify Block</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14922</td>
      <td>True</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-71193b63-0f32-4744-9859-bc6cce0fd0d2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-71193b63-0f32-4744-9859-bc6cce0fd0d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-71193b63-0f32-4744-9859-bc6cce0fd0d2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b785a458-a40a-426d-b648-4e22e40cf6cd">
  <button class="colab-df-quickchart" onclick="quickchart('df-b785a458-a40a-426d-b648-4e22e40cf6cd')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b785a458-a40a-426d-b648-4e22e40cf6cd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
print(f"To maximize magnitude and therefore value, create {num_potions} potions of the {len(to_make_df)} unique types listed above for a total magnitude of {total_magnitude}.")
```

    To maximize magnitude and therefore value, create 76 potions of the 42 unique types listed above for a total magnitude of 3905.

