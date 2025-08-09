Source codes and libraries of my gradute thesis

# Tutorial

* Create a TSMCN65 library in Cadence Virtuoso.
* Include all cells in Cadence_library folder (DATN, DATN_TEST, StrongArmLatch, Thesis).
* Make sure to create netlist file (ADE L > Simulation > Netlist > Create).
* Copy OTA_ESOA.py and comparator_ESOA.py to OTA and Comparator folder respectively.
* Navigate to the folder and run command: 
    ``` bash
    python OTA_ESOA.py
    ```
    if you want to optimize OTA module or 
    ``` bash
    python comaprator_ESOA.py
    ```      
    if you want to optimize compator module.
* When the optimization progress is finished, you will see a csv file which contains all the result of all epochs.
