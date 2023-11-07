Readme

The Python script works without any shell interaction, just fill the requested variables.

The variable Sample admits one of the following:
'InAs': for InAs samples
'InSb' for InSb sample TMW12007 from Engineering (Maurizio's)
'AlInSb' for AlInSb sample TMW12013 from engineering (Maurizio's)

Each of this samples has its own thickness and substrate refractive index If necessary to add a new sample simply open the file ConductivityModels.py and inside the class Sample add the desired model following the template.

The variable flip is set to False if both the data sets have been taken from positive position to negative, in opposite case flip has to be set True.
I never tried hybrid cases.

The variable save saves the outcome in a .txt file in the path and name defined by the variable Res, the file has an header, anyway the data are saved as following:

frequency(THz), real part of the average value of photoconductivity(1 / (Ohm*m), std dev of the real part of photoconductivity(1/(Ohm*m)), imaginary part of the average value of photoconductivity(1 / (Ohm*m), std dev of the imaginary part of photoconductivity(1/(Ohm*m))

The separator is a tab '\t'


