"""
Generate the `dfi_subset` for differnet models
"""
from collections import OrderedDict

# Use representative motifs for each experiment specified in config.py
from config import experiments

models_dir = '/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/output'

def dict2yaml_str(d):
    return ";".join([f'{k}={v}'.replace(" ", "").replace("'", '"')
                     for k, v in d.items()])

def get_kwargs(wildcards):
    exp = wildcards.exp
    exp_kwargs = experiments[exp]
    exp_kwargs['exp'] = exp
    return dict2yaml_str(exp_kwargs)


rule all:
    input:
        expand(f"{models_dir}/{{exp}}/deeplift/dfi_subset.parq",
               exp=list(experiments),
               ),

rule gen_dfi_subset:
    """
    Generate the dfi_subset file
    """
    output:
        parq = f"{models_dir}/{{exp}}/deeplift/dfi_subset.parq"
    params:
        imp_score = lambda wildcards: experiments[wildcards.exp]['imp_score'],
        motifs = lambda wildcards: dict2yaml_str(experiments[wildcards.exp]['motifs']),
        # annotate dfi with profile only for the default experiment
        profile = lambda wildcards: ('--append-profile' if (wildcards.exp == "nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE" or
                                                           wildcards.exp.startswith("nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,"))
                                                        else '')
    shell:
        """
        python generate_dfi_subset.py {wildcards.exp} --imp-score {params.imp_score} --motifs '{params.motifs}' {params.profile}
        """
