Data Requirements for Networks Training:
        
*All the process(=>) below have been done by dataPreProcessing.py (except the part of 'Video => Photo => Skeleton Coordinations' in 1.2 output)
*This folder provides raw data taken from the datasets and processed data that can be used for training networks.
    
    
1. Bowing attack network : 
    Data source : URMP dataset http://www2.ece.rochester.edu/projects/air/projects/URMP.html
        
    1.1 input: Mel-spectrograms
        Audio("./Violin_Audio_Wav/", directly from URMP) => Mel-spectrograms("./Violin_Audio_Csv/")
        
    1.2 output: Bowing Attack Labels
        Video(directly from URMP) => Photos => Skeleton Coordinations extracted by Openpose("./Violin_Skeleton_Json_unfixed/") => 
        Normalized and Smoothed Skeleton Coordination("./Violin_Skeleton_Csv_fixed/") => Bowing Attack Labels("./Violin_BowingAttacks_Txt/")
    

2. Arousal Network : data requirements
        Data source : DEAM dataset http://cvml.unige.ch/databases/DEAM/
        
    2.1 input: Mel-spectrograms
        Audio("./Emotion_Audio_Wav/", directly from DEAM) => Mel-spectrograms("./Emotion_Audio_Csv/")
        
    2.2 output: Arousal Attack Labels
        Arousal Attack Labels("./Emotion_Arousal_Csv/", directly from DEAM)
               
