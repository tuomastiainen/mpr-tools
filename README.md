# MPR-tools

Multi-probe roundness measurement algorithms and signal generator.

![Multi probe roundness measurement](img/fourpoint.png?raw=true "Multi probe roundness measurement")

From at least three probe signals, multi probe roundness measurement methods can separate the center point motion and roundness profile, which are both included in the probe signals. The methods are especially needed for precise measurements of roundness when a precision spindle cannot be used, for example when measuring large flexible rotors.

## Algorithms
All of the functions take the raw probe signals as input and provide roundness profile Fourier coefficients as output. From the complex Fourier coefficients, the roundness profile can be reconstructed with get_roundness_profile function.

### Three-probe
ozono_f_coeff

Calculates roundness profile from three probe signals.

### Least squares method
jansen_roundness_f_coeff

Calculates roundness profile from four probe signals using a least squares minimization.

### Redundant diameter method
hybrid_f_coeff

Calculates roundness profile from four probe signals using three-probe calculation for odd coefficients and diameter variation profile for even coefficients.


## Getting started
Install requirements and run mpr_tools, it will generate signals and display the calculated roundness profile.

```
python mpr_tools.py
```


## Authors

* Tuomas Tiainen 2019-2020

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

