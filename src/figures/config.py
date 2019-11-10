experiments = {
    # ChIP-nexus
    "nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE": {
        "imp_score": "profile/wn",
        "motifs": {
            "Oct4-Sox2": 'Oct4/m0_p0',
            "Oct4": 'Oct4/m0_p1',
            "Oct4-Oct4": 'Oct4/m0_p6',
            "B-Box": 'Oct4/m0_p5',
            "Sox2": 'Sox2/m0_p1',
            "Nanog": 'Nanog/m0_p1',
            "Nanog-partner": 'Nanog/m0_p4',
            # "Nanog-mix": 'Nanog/m0_p5',
            "Zic3": 'Nanog/m0_p2',
            "Klf4": 'Klf4/m0_p0',
            "Klf4-Klf4": 'Klf4/m0_p5',
            'Essrb': 'Oct4/m0_p16'
        },
    },
    "nexus,gw,OSNK,1,0,0,FALSE,same,0.5,64,25,0.001,9,FALSE": {
        "imp_score": 'class/pre-act',
        "motifs": {"Oct4-Sox2": "Oct4/m0_p0",
                   "Oct4": "Oct4/m0_p9",
                   "Sox2": "Sox2/m0_p3",
                   "Nanog": "Nanog/m0_p2",
                   "Klf4": "Klf4/m0_p0"},
    },
    # ChIP-seq
    "seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE,TRUE": {
        "imp_score": "profile/wn",
        "motifs": {"Oct4-Sox2": "Oct4/m0_p0",
                   "Oct4": "Oct4/m0_p5",
                   "Sox2": "Sox2/m0_p1",
                   "Nanog": "Nanog/m0_p1"
                   }
    },
    "seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE": {
        "imp_score": "profile/wn",
        "motifs": {"Oct4-Sox2": "Oct4/m0_p0",
                   "Oct4": "Oct4/m0_p1",
                   "Sox2": "Sox2/m0_p1",
                   "Nanog": "Nanog/m0_p1",
                   }
    },
    "seq,gw,OSN,1,0,0,FALSE,same,0.5,64,50,0.001,9,FALSE": {
        "imp_score": "class/pre-act",
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p2",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p3",
            "Zic3": "Nanog/m0_p1",
        },
    },
    #     # Joint
    #     "nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE": {
    #         "imp_score": 'profile/wn',
    #         "motifs": {
    #             "Oct4-Sox2": "Oct4/m0_p0",
    #             "Oct4": "Oct4/m0_p5",
    #             "Sox2": "Sox2/m0_p1",
    #             "Nanog": "Nanog/m0_p1",
    #         }
    #     },
    #     "seq,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE": {
    #         "imp_score": 'profile/wn',
    #         "motifs": {
    #             "Oct4-Sox2": "Oct4/m0_p0",
    #             "Oct4": "Oct4/m0_p3",
    #             "Sox2": "Sox2/m0_p1",
    #             "Nanog": "Nanog/m0_p2",
    #         }
    #     },
    'seq,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE,TRUE,0': {
        "imp_score": 'profile/wn',
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p2",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p1",
        }
    },
    'nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,TRUE,0': {
        "imp_score": 'profile/wn',
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p1",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p1",
        }
    },
    # not augmented
    'nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,0': {
        "imp_score": 'profile/wn',
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p3",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p0",
        }
    },
    # not augmented
    'nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,0': {
        "imp_score": 'profile/wn',
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p3",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p0",
        }
    },
    # Not bias corrected
    "nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE-2": {
        "imp_score": "profile/wn",
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p4",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p0",
            "Klf4": "Klf4/m0_p0",
        }
    },
    "seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE": {
        "imp_score": "profile/wn",
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p2",
            "Sox2": "Sox2/m0_p1",
            "Nanog": "Nanog/m0_p2",
        }
    },
    # basset
    "binary-basset,nexus,gw,OSNK,0.5,64,0.001,FALSE,0.6": {
        "imp_score": "class/pre-act",
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p5",
            "Sox2": "Sox2/m0_p3",
            "Nanog": "Nanog/m0_p2",
            "Klf4": "Klf4/m0_p0",
        }
    },
    "factorized-basset,nexus,gw,OSNK,0.5,64,0.001,FALSE,0.5": {
        "imp_score": "class/pre-act",
        "motifs": {
            "Oct4-Sox2": "Oct4/m0_p0",
            "Oct4": "Oct4/m0_p5",
            "Sox2": "Sox2/m0_p2",
            "Nanog": "Nanog/m0_p2",
            "Klf4": "Klf4/m0_p0",
        }
    },
    'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,1': {'imp_score': 'profile/wn',
      'motifs': {'Oct4-Sox2': 'Oct4/m0_p0',
       'Oct4': 'Oct4/m0_p1',
       # 'Oct4-Oct4': 'Oct4/m0_p0',
       'B-Box': 'Oct4/m0_p5',
       'Sox2': 'Oct4/m0_p4',
       'Nanog': 'Nanog/m0_p0',
       'Nanog-partner': 'Nanog/m0_p4',
       'Zic3': 'Nanog/m0_p3',
       'Klf4': 'Klf4/m0_p0',
       'Klf4-Klf4': 'Klf4/m0_p4',
       'Essrb': 'Klf4/m0_p3'}},
     'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,2': {'imp_score': 'profile/wn',
      'motifs': {'Oct4-Sox2': 'Sox2/m0_p0',
       'Oct4': 'Oct4/m0_p7',
       # 'Oct4-Oct4': 'Oct4/m0_p0',
       'B-Box': 'Nanog/m0_p14',
       'Sox2': 'Sox2/m0_p1',
       'Nanog': 'Nanog/m0_p1',
       'Nanog-partner': 'Oct4/m0_p7',
       'Zic3': 'Klf4/m0_p2',
       'Klf4': 'Klf4/m0_p0',
       'Klf4-Klf4': 'Klf4/m0_p6',
       'Essrb': 'Oct4/m0_p8'}},
     'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,3': {'imp_score': 'profile/wn',
      'motifs': {'Oct4-Sox2': 'Klf4/m0_p2',
       'Oct4': 'Oct4/m0_p0',
       # 'Oct4-Oct4': 'Oct4/m0_p6',
       'B-Box': 'Oct4/m0_p5',
       'Sox2': 'Sox2/m0_p1',
       'Nanog': 'Nanog/m0_p2',
       'Nanog-partner': 'Sox2/m0_p9',
       'Zic3': 'Nanog/m0_p3',
       'Klf4': 'Klf4/m0_p6',
       'Klf4-Klf4': 'Klf4/m0_p6',
       'Essrb': 'Oct4/m0_p7'}},
     'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,4': {'imp_score': 'profile/wn',
      'motifs': {'Oct4-Sox2': 'Sox2/m0_p0',
       'Oct4': 'Oct4/m0_p0',
       # 'Oct4-Oct4': 'Oct4/m0_p9',
       'B-Box': 'Oct4/m0_p4',
       'Sox2': 'Sox2/m0_p1',
       'Nanog': 'Nanog/m0_p1',
       'Nanog-partner': 'Nanog/m0_p2',
       'Zic3': 'Nanog/m0_p4',
       'Klf4': 'Klf4/m0_p0',
       'Klf4-Klf4': 'Klf4/m0_p2',
       'Essrb': 'Klf4/m0_p5'}},
     'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,FALSE,5': {'imp_score': 'profile/wn',
      'motifs': {'Oct4-Sox2': 'Sox2/m0_p0',
       'Oct4': 'Oct4/m0_p2',
       # 'Oct4-Oct4': 'Oct4/m0_p0',
       'B-Box': 'Oct4/m0_p6',
       'Sox2': 'Sox2/m0_p1',
       'Nanog': 'Nanog/m0_p7',
       'Nanog-partner': 'Nanog/m0_p6',
       'Zic3': 'Nanog/m0_p2',
       'Klf4': 'Klf4/m0_p0',
       'Klf4-Klf4': 'Klf4/m0_p5',
       'Essrb': 'Klf4/m0_p3'}}
}
