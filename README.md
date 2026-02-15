{% include syntax.html type="basic-sub" syntax-id="unordered-lists" %}
# HZGP
Horizontal gravel pack calculation of alpha wave height and minimum transport slurry rate.

This uses a model that can be modified to accept various assembly dimensions and slurry properties to calculate the
velocity of the slurry in the channel above the alpha wave sand bed.  The code uses the calculation of velocity
to vary the solids concentration in the slurry, until the flowing pressure gradients in the wash pipe by screen annulus is
the same as pressure gradient in the channel above the sand bed.

The detailed description of the calculations made by HZGP is in SPE-230528-MS presented at the 2026 SPE International Conference and Exhibition on Formation Damage Control.

The primary feature of this calculation is that it can estimate the minimum surface slurry rate that can transport gravel
pack sand along the open hole horizontal.

## To set up HZGP
    *   open a terminal in this folder
    *   create a new environment 
        py -m venv .\.venv
    *   activate the environment
        .\.venv\Scripts\activate.ps1
    *   install the packages in requirements.txt
        pip install -r requirements.txt
    *   once pip has installed the necessary packages, to run HZGP, run the powershell script
        .\hzgp.ps1
Python will start a local server and the default brower.  It will take a few seconds but the browser should display the HZGP screen. You may need to refresh the browser a couple of times. It takes python sometime to start the program and the server that displays the input and output screens.

## To use HZGP
HZGP is very simplistic. The left side has the inputs and controls. The right side has a results chart and a summary results table of  the analyses. 

### To start HZGP 
    *   as described above, from the powershell window running in the installation folder:
        .\hzgp.ps1
    *   from the file explorer, right click on the script "hzgp.ps1" select "Run with PowerShell
    
The script starts the default browser, but starting python may take a few seconds. You may need to refresh before HZGP is displayed.

### HZGP's input and results tabs

The "slurry" tab has two buttons:
    *   "Run" this adds a line to the chart representing the current inputs and a line to the table summarizing the minimum slurry rate Q_min and maximum bed depth, h_max for the current slurry and BHA inputs.
    *   "Clear" this clears the chart and table and then displace a line on the chart and row in the table representing the results for the current slurry and BHA.
Several models for the critical velocity can be selected in the dropdown in the slurry tab.  These are:

    * models 1 and 2 - these are discussed in SPE-230528-MS
    * Oroskar and Turian 
    * Linear - a model developed from a linear regression of ~ 300 test results 
    * Constant - a constant V_crit that does not change with bed height or slurry properties
Note the enthusiastic user can also add their V_crit model by simply changing or adding to the code in hzgp.py and minimumVc.py

Several charts can be displaced in the Graph tab.

    * Slurry rate vs bed height (rate-height)
    * Slurry rate vs pressure gradient above a stable alpha wave bed (rate- dp/dx alpha)
    * Slurry rate vs pressure gradient in fully packed hole - or beta wave (rate - dp/dx beta)
    * Slurry rate vs pressure gradient of the return flow inside the wash pipe (rate- dp/dx washpipe id)
    * Slurry rate can be charted either in gpm or bpm

The inputs are generally straight forward. The BHA (screen, washpipe and OH diameter) dimensions are input in inches. The units for slurry inputs are given in the label. The one possible confusion are the inputs marked "DR" for drag reduction.  Low dose polymers can reduce flowing friction - the so called Tom's effect.  The number in these inputs represent the effect of a drag reducer slurry additive, if present. DR should be 0 if there is no additive. DR is 1 when a slurry additive achieves the maximum possible drag reduction.  The maximum is assumed to be given by Virk's aysptotic drag reduction, formula.