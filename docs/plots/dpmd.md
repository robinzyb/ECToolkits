# Plot DPMD Test Data

Nowadays, I found that many studies for the simulations of interfaces use DeePMD-kit.
In the end, users will need to test the accuracy of the machine learning potentials.
Here, I implemented a simple script to plot the results of the tests.

Assume users have had two dptest files from DeePMD-kit, called `dptest.e.out` and `dptest.f.out`.
Now we import the plot function from `ECToolkits`
```python
from ectoolkits.plots.dpmd import plot_dptest

e_file = "./dptest.e.out"
f_file = "./dptest.f.out"
save_png = "dptest.png"


fig = plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, frc_comp=True, return_err=False)

fig, rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz = plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, frc_comp=True, return_err=True)
fig, rmse_e, mae_e, rmse_f, mae_f = plot_dptest(e_file=e_file, f_file=f_file, save_name=save_png, frc_comp=False, return_err=True)
```
With the above codes, users can get the plot of the test results.

![test](./figures/dptest_example.png)