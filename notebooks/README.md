# The following notebooks illustrate the reconstruction pipeline:

### full_pipeline.ipynb (runnable if the data is provided)

- This notebook illustrate how the reconstruction is applied from begining

### full_pipeline_without_butter.ipynb (runnable if the data is provided)

- Butter filter is removed to see its effects:

	- ICA runs slower
	- Idetifying the fetal/maternal components harder
	- If fetal component identifiable, then HOSD works still nice, no need of butter

### full_pipeline_first_HOSD.ipynb (runnable if the data is provided)

- Out of curiosity, first HOSD filtering is applied to every channel of the raw data

	- It completely clears the fetal signal
	- Probably, since maternal >> fetal, filter thinks fetal component as noise; thus, clears it

### eval_p45.ipynb (cannot be run)

- Secondly developed python notebook to show that converted python code actually works the same as the matlab script

### colab_basic_analysis.ipynb (cannot be run)

- Firstly developed python notebook to understand how ICA works and the structure of cardiac signals
