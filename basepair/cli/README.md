## Model training pipeline

1. Write a generic dataloader
  - fasta_file
  - bw's (with tasks)
  - peaks.bed
    - combination strategy: intersect, union?
2. Infer the right model from the data
3. Train the model. Save:
  - model.h5
  - loss-curves.png
  - history.csv
4. Evaluate the model
  - compute all the possible metrics
5. Run Modisco:
  - run it for different use-cases
    - 'max profile' each task
    - 'counts' each task 
6. Plot all results
  - one ipynb
    - loss curves
    - modisco results
    - few example profiles
  - for each plots, save the results


--------------------------------------------
Directory structure:
```
data/  # input data
  task_name/
    encode_pipeline/
	   ... # output of the encode pipeline

    # softlink the important files here to stay agnostic to the pipeline-naming
    peaks.bed
    5prime.counts.pos.bw
    5prime.counts.neg.bw
models/
  0/
    model.h5
    loss-curves.png
    history.csv
	eval/
	  metrics.json
	  profile_metrics.tsv
	  plots/
	    counts.{split}.{task}.png
  1/
   ...
modisco/  # modisco results
  valid/
	modisco.h5
    distances.npy
    included_samples.npy
  train/
    modisco.h5
    distances.npy
	included_samples.npy

figures/  # all the produced figures
results.ipynb 
```
