# --- Opening Positions ---
# A list of (FEN, Name) tuples to start games from.
POSITIONS = [
    # Classical and well-known
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Italian Game"),
    ("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Defense: Najdorf Variation"),
    ("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", "Queen's Gambit Declined"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Ruy Lopez"),
    ("rnbqkbnr/pppp2pp/4p3/5p2/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3", "French Defense"),
    ("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3", "Caro-Kann Defense"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3", "English Opening"),
    ("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 1 6", "King's Indian Defense"),
    ("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4", "Nimzo-Indian Defense"),
    ("rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3", "Scandinavian Defense"),

    # Gambits and sharp lines
    ("rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R w KQkq - 0 3", "King's Gambit Accepted"),
    ("rnbqkb1r/pppp1ppp/8/4p1n1/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 4", "Budapest Gambit"),
    ("rnbqkbnr/ppp2ppp/8/4P3/2Pp4/8/PP2PPPP/RNBQKBNR w KQkq - 0 4", "Albin Counter-Gambit"),
    ("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "Center Game"),
    ("rnbqkbnr/pppp1ppp/8/8/3pP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3", "Danish Gambit Accepted"),
    ("rnbqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 2", "Vienna Game"),
    ("rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", "Queen's Gambit Accepted"),

    # Hypermodern and flank openings
    ("rnbqkb1r/p2ppppp/5n2/1ppP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4", "Benko Gambit"),
    ("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "Englund Gambit"),
    ("rnbqkb1r/pp3ppp/4p3/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5", "Tarrasch Defense"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R w KQkq - 2 3", "London System"),
    ("rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "Dutch Defense"),
    ("rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4", "Benoni Defense"),
    ("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5NP1/PPP1PP1P/RNBQKB1R b KQkq - 0 3", "King's Indian Attack"),
    ("rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1", "Bird's Opening"),

    # Esoteric and unusual systems
    ("rnbqkbnr/pppppppp/8/8/6P1/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1", "Grob's Attack"),
    ("rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1", "Nimzowitsch-Larsen Attack"),
    ("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1", "Polish Opening (Sokolsky)"),
    ("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2", "Zukertort Opening"),
    ("rnbqkb1r/pppppppp/5n2/8/3P1B2/2N5/PPP1PPPP/R2QKBNR b KQkq - 2 2", "Jobava London System"),
    ("rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 2 5", "Colle System"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b KQkq - 1 2", "Bongcloud Attack"),
    ("rnbqk2r/pppp1ppp/5n2/2b1p2Q/4P3/2N5/PPPP1PPP/R1B1KBNR w KQkq - 4 4", "Wayward Queen Attack"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2P2N2/PP1P1PPP/RNBQKB1R w KQkq - 0 3", "Ponziani Opening"),
    ("rnbqkb1r/pppppppp/5n2/6B1/3P4/8/PPP2PPP/RN1QKB1R b KQkq - 1 2", "Torre Attack"),
    ("rnbqkbnr/pp2pppp/8/3p4/2P5/5N2/PP1P1PPP/RNBQKB1R w KQkq - 0 3", "Réti Opening"),
    ("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "Elephant Gambit"),

    # Additional Openings
    ("rnbqkb1r/ppp1pppp/5n2/6B1/3p4/3P1N2/PPP1PPPP/RN1QKB1R w KQkq - 0 4", "Trompowsky Attack"),
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 4", "Evans Gambit"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq - 0 2", "Sicilian Defense: Alapin Variation"),
    ("rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3", "Slav Defense"),
    ("rnbqkb1r/pp2pp1p/5np1/3p4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6", "Grünfeld Defense"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "Scotch Game"),
    ("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2", "Bishop's Opening"),
    ("rnbqkbnr/pppp1p1p/8/4p1p1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "Latvian Gambit"),
    
    # Final 5 to make 50
    ("rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Sicilian Defense: Dragon Variation"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4", "Four Knights Game"),
    ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Petrov's Defense"),
    ("rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4", "Queen's Indian Defense"),
    ("rnbqkbnr/pppppp1p/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Modern Defense"),
]
