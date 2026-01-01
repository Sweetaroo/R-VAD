SCOPE_PROFILES = {
    "nut": {
        "allergen_nut": {
            "queries": [
                "nut",
                "peanut",
                "hazelnut",
                "almond",
                "walnut",
                "tree nut",
            ],
            "claim_queries": ["nut pieces", "peanut pieces"],
            "claim_triggers": [
                "nut-free",
                "no nuts",
                "safe for nut allergy",
                "allergy-safe (nuts)",
            ],
            "min_confidence": 0.5,
        }
    },
    "allergens4": {
        "allergen_nut": {
            "queries": [
                "nut",
                "peanut",
                "hazelnut",
                "almond",
                "walnut",
                "tree nut",
            ],
            "claim_queries": ["nut pieces", "peanut pieces"],
            "claim_triggers": [
                "nut-free",
                "no nuts",
                "safe for nut allergy",
                "allergy-safe (nuts)",
            ],
            "min_confidence": 0.5,
        },
        "allergen_seafood": {
            "queries": ["shrimp", "crab", "lobster", "shellfish", "seafood"],
            "claim_queries": ["shrimp pieces", "shellfish pieces"],
            "claim_triggers": [
                "seafood-free",
                "no shellfish",
                "safe for shellfish allergy",
            ],
            "min_confidence": 0.5,
        },
        "allergen_dairy": {
            "queries": ["milk", "cheese", "butter", "cream", "yogurt", "dairy"],
            "claim_queries": ["milk residue", "cheese pieces"],
            "claim_triggers": [
                "dairy-free",
                "no milk",
                "lactose-free",
                "safe for dairy allergy",
            ],
            "min_confidence": 0.5,
        },
        "allergen_gluten": {
            "queries": ["bread", "pasta", "noodles", "wheat", "flour", "gluten"],
            "claim_queries": ["wheat flour", "gluten traces"],
            "claim_triggers": [
                "gluten-free",
                "no wheat",
                "celiac-safe",
                "safe for gluten allergy",
            ],
            "min_confidence": 0.7,
        },
    },
    "allergens4_plus": {
        "allergen_nut": {
            "queries": [
                "nut",
                "peanut",
                "hazelnut",
                "almond",
                "walnut",
                "tree nut",
            ],
            "claim_queries": ["nut pieces", "peanut pieces"],
            "claim_triggers": [
                "nut-free",
                "no nuts",
                "safe for nut allergy",
                "allergy-safe (nuts)",
            ],
            "min_confidence": 0.5,
        },
        "allergen_seafood": {
            "queries": ["shrimp", "crab", "lobster", "shellfish", "seafood"],
            "claim_queries": ["shrimp pieces", "shellfish pieces"],
            "claim_triggers": [
                "seafood-free",
                "no shellfish",
                "safe for shellfish allergy",
            ],
            "min_confidence": 0.5,
        },
        "allergen_dairy": {
            "queries": ["milk", "cheese", "butter", "cream", "yogurt", "dairy"],
            "claim_queries": ["milk residue", "cheese pieces"],
            "claim_triggers": [
                "dairy-free",
                "no milk",
                "lactose-free",
                "safe for dairy allergy",
            ],
            "min_confidence": 0.5,
        },
        "allergen_gluten": {
            "queries": ["bread", "pasta", "noodles", "wheat", "flour", "gluten"],
            "claim_queries": ["wheat flour", "gluten traces"],
            "claim_triggers": [
                "gluten-free",
                "no wheat",
                "celiac-safe",
                "safe for gluten allergy",
            ],
            "min_confidence": 0.7,
        },
    },
    "regulated_demo": {
        "regulated_cannabis": {
            "queries": ["cannabis", "marijuana", "THC", "weed", "edible gummy"],
            "claim_queries": ["cannabis residue", "THC gummy"],
            "claim_triggers": ["drug-free", "no THC", "non-psychoactive"],
            "min_confidence": 0.5,
        },
        "regulated_medication": {
            "queries": ["pill", "capsule", "medicine bottle", "prescription"],
            "claim_queries": ["pill bottle", "capsule pieces"],
            "claim_triggers": ["no medication", "drug-free"],
            "min_confidence": 0.5,
        },
    },
}
