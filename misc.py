# sox effects when splitting long audio
splitter_sox_effects: List[str] = [
        ["norm"],  # normalize audio

        # isolate voice frequency
        # ["highpass", "-1", "100"],
        # ["lowpass", "-1", "3000"],
        # -2 is for a steeper filtering: removes high frequency and very low ones
        # ["highpass", "-2", "50"],
        # ["lowpass", "-2", "5000"],
        # ["norm"],  # normalize audio

        # max silence should be 3s
        ["silence", "-l", "1", "0", "1%", "-1", "3.0", "1%"],

        ["norm"],
        ]

