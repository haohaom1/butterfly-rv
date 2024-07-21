## Installment
pip install -r requirements.txt

## running the code
``` 
python app.py
```

Then paste the local link into the browser. The default should be http://127.0.0.1:8050/

# butterfly-rv

This is an interactive visualization tool that determines the richest & cheapest flies given user inputs and visually displays the results

### Cleaning the given yields data and extracting the benchmark yields

* In the EDA.ipynb, I explore the given dataset and use it to construct a par curve. I noticed that only current bonds were given in the original data, which means I can only build a par curve dating back to the last two years, as anytime before that any bonds in the 2y sector would have all matured and therefore not exist in the given dataset.
* I took a very simplistic approach of plotting all given bonds based on time to maturity, then running a cubic spline to interpolate the benchmark points. In practice, I would use a well researched method of building a par curve, which involve using duration instead of time to maturity, modelling the frontend using treasury bill inputs, and weighting the on-the-run issues more in the spline interpolation.

### Building out the analysis

* I used plotly dash as the frontend, as it provides an easy interface for the user to interact with the data. 
* My approach was to find butterflies that are hedged for the duration and curve, thereby leaving a mean reverting butterfly. By looking at how much it is from its mean, I can determine how rich or cheap the weighted butterfly is. 

### Using the tool

* The user first selects the date range they wish to use, as well as the analytical method (PCA or regression). The tool then returns the top 3 richest and cheapest flies after neutralizing for the duration & curve (PC1 & PC2). This weighted fly is assumed to be mean reverting as the first and second order terms are hedged from construction. 
* The user can also examine individual flies by selecting the left wing, belly, and right wing of the fly, and using the given date ranges and analytical method displays the regression to PC1 and PC2, as well as a historical time series of the PC1 and PC2 neutral fly. Certain statistics are also displayed on the weighted fly.

### Further improvements

* One metric I would also look at in practice is the roll / carry of such butterfly at inception, as this can help determine the attractiveness of certain butterflies over others. 
* Another improvement would be to backtest this strategy of neutralizing duration / curve over history, as a primary assumption is that beta to PC1 & PC2 is stable over time, and running a backtest or a rolling beta / pca can help determine that.
* A third improvement would be to use the second of freedom to maximize / minimize another metric, instead of simply neutralizing curve / PC2. As doing this project, I noticed that often just the duration / PC1 is enough to explain most of the variability in a fly. Therefore, as butterflies innately have two degrees of freedom, we can use one to hedge out duration, and use the second one to minimize vol or maximize carry for example.
