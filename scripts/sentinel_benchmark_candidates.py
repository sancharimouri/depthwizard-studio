# Candidate locations for the India-wide Sentinel-2 benchmark dataset, one
# list per terrain category. Coordinates are approximate city/region centers
# chosen by geographic reasoning (Himalayan/Western-Ghats hill towns for
# "hilly", major built-up cities for "urban", known agricultural belts for
# "agricultural", and delta/estuary/backwater stretches for "coastal") — not
# arbitrary picks. Existing production regions (Darjeeling, Kolkata,
# Bardhaman, Sundarbans) are excluded since they're already in the live site.

CANDIDATES = {
    "hilly": [
        ("gangtok", "Gangtok, Sikkim", 27.3389, 88.6065),
        ("kalimpong", "Kalimpong, WB", 27.0669, 88.4711),
        ("kurseong", "Kurseong, WB", 26.8814, 88.2836),
        ("shimla", "Shimla, HP", 31.1048, 77.1734),
        ("manali", "Manali, HP", 32.2432, 77.1892),
        ("dharamshala", "Dharamshala, HP", 32.2190, 76.3234),
        ("dehradun", "Dehradun, Uttarakhand", 30.3165, 78.0322),
        ("mussoorie", "Mussoorie, Uttarakhand", 30.4598, 78.0664),
        ("nainital", "Nainital, Uttarakhand", 29.3919, 79.4542),
        ("almora", "Almora, Uttarakhand", 29.5892, 79.6467),
        ("shillong", "Shillong, Meghalaya", 25.5788, 91.8933),
        ("kohima", "Kohima, Nagaland", 25.6751, 94.1086),
        ("ooty", "Ooty (Nilgiris), TN", 11.4064, 76.6932),
        ("munnar", "Munnar (Western Ghats), Kerala", 10.0889, 77.0595),
    ],
    "urban": [
        # These six already have real, previously-acquired Sentinel-2 tiles
        # in data/sentinel2/{name}/ (from the Method-3 building-footprint
        # track) — reused here rather than re-downloaded. Centers computed
        # from each tile's actual raster bounds.
        ("bengaluru", "Bengaluru, Karnataka", 12.9716, 77.5946),
        ("delhi", "Delhi", 28.6139, 77.2090),
        ("hyderabad", "Hyderabad, Telangana", 17.3850, 78.4867),
        ("jaipur", "Jaipur, Rajasthan", 26.9124, 75.7873),
        ("kochi_city", "Kochi, Kerala", 9.9312, 76.2673),
        ("mumbai", "Mumbai, Maharashtra", 19.0760, 72.8777),
        # Backups, only queried/downloaded if one of the above fails coverage.
        ("chennai", "Chennai, TN", 13.0827, 80.2707),
        ("pune", "Pune, Maharashtra", 18.5204, 73.8567),
    ],
    "agricultural": [
        ("ludhiana", "Ludhiana, Punjab", 30.9010, 75.8573),
        ("sangrur", "Sangrur, Punjab", 30.2458, 75.8421),
        ("bathinda", "Bathinda, Punjab", 30.2110, 74.9455),
        ("karnal", "Karnal, Haryana", 29.6857, 76.9905),
        ("hisar", "Hisar, Haryana", 29.1492, 75.7217),
        ("meerut", "Meerut, UP", 28.9845, 77.7064),
        ("fatehpur", "Fatehpur (Ganga plain), UP", 25.9308, 80.8155),
        ("guntur", "Guntur, AP", 16.3067, 80.4365),
        ("kurnool", "Kurnool, AP", 15.8281, 78.0373),
        ("nizamabad", "Nizamabad, Telangana", 18.6725, 78.0941),
        ("kota", "Kota (Chambal command area), Rajasthan", 25.2138, 75.8648),
        ("dewas", "Dewas (Malwa soybean belt), MP", 22.9623, 76.0534),
        ("vidisha", "Vidisha, MP", 23.5251, 77.8081),
        ("raichur", "Raichur, Karnataka", 16.2076, 77.3463),
        ("erode", "Erode (Cauvery belt), TN", 11.3410, 77.7172),
    ],
    "coastal": [
        ("bhitarkanika", "Bhitarkanika/Kendrapara delta, Odisha", 20.7167, 86.9167),
        ("puri", "Puri coast, Odisha", 19.8135, 85.8312),
        ("chilika", "Chilika Lake, Odisha", 19.7000, 85.3200),
        ("kakinada", "Kakinada (Godavari delta), AP", 16.9891, 82.2475),
        ("amalapuram", "Amalapuram (Godavari delta), AP", 16.5786, 82.0086),
        ("machilipatnam", "Machilipatnam (Krishna delta), AP", 16.1875, 81.1389),
        ("nagapattinam", "Nagapattinam (Cauvery delta), TN", 10.7672, 79.8449),
        ("vedaranyam", "Vedaranyam, TN", 10.3765, 79.8517),
        ("vembanad", "Vembanad backwaters, Kerala", 9.4981, 76.3388),
        ("digha", "Digha coast, WB", 21.6270, 87.5100),
        ("bharuch", "Bharuch (Narmada estuary), Gujarat", 21.7051, 72.9959),
        ("kutch", "Gulf of Kutch coastal fringe, Gujarat", 23.2420, 68.9685),
        ("ratnagiri", "Ratnagiri (Konkan coast), Maharashtra", 16.9902, 73.3120),
        ("goa_estuary", "Mandovi-Zuari estuary, Goa", 15.4909, 73.8278),
        ("mangalore", "Netravati-Gurupura estuary, Karnataka", 12.8703, 74.8806),
    ],
}
