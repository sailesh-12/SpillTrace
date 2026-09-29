"""Reference gazetteer for Indian waters: environmentally / economically sensitive sites, major ports and the
Indian Coast Guard Maritime Rescue Co-ordination Centres (MRCCs).

IMPORTANT: coordinates are APPROXIMATE reference points (site centroids, a few km accuracy) compiled for this
decision-support prototype. They are not official Environmental Sensitivity Index (ESI) maps. For operational
use, replace this module with the official ESI / INCOIS coastal sensitivity layers.

Sensitivity weights (1-10) follow the spirit of NOAA ESI ranking: mangroves, coral reefs, turtle-nesting beaches
and lagoons are most sensitive; critical seawater intakes (power plants) are high because oil ingress forces
shutdowns; ports and fishing harbours are economic receptors.
"""
from __future__ import annotations

# (id, name, type, lon, lat, radius_km, weight, receptor_note)
SITES: list[tuple[str, str, str, float, float, float, int, str]] = [
    ("gulf_kutch_mnp", "Gulf of Kutch Marine National Park", "coral_mangrove", 69.80, 22.45, 25, 10, "coral reefs, mangroves"),
    ("veraval", "Veraval fishing harbour", "fishing", 70.37, 20.90, 8, 7, "one of India's largest fishing harbours"),
    ("kandla", "Deendayal (Kandla) Port", "port", 70.22, 23.00, 8, 6, "major port, oil terminals"),
    ("tarapur", "Tarapur power station seawater intake", "intake", 72.65, 19.83, 5, 9, "nuclear/thermal plant cooling-water intake"),
    ("thane_creek", "Thane Creek Flamingo Sanctuary", "mangrove", 72.98, 19.10, 8, 9, "mangroves, flamingo habitat"),
    ("mumbai_port", "Mumbai Port / JNPT", "port", 72.90, 18.93, 10, 6, "major ports, dense shipping"),
    ("alibag", "Alibag–Murud beaches", "tourism", 72.87, 18.55, 10, 6, "tourism beaches, fishing villages"),
    ("ratnagiri", "Ratnagiri fishing harbour", "fishing", 73.28, 16.99, 8, 7, "fishing harbour"),
    ("malvan", "Malvan Marine Sanctuary", "coral", 73.45, 16.05, 10, 9, "coral patches, marine sanctuary"),
    ("goa", "Goa beaches (Mormugao)", "tourism", 73.78, 15.45, 20, 7, "tourism coastline, Mormugao Port"),
    ("netrani", "Netrani Island", "coral", 74.32, 14.02, 5, 8, "coral reef"),
    ("mangaluru", "New Mangalore Port", "port", 74.80, 12.92, 8, 6, "major port"),
    ("lakshadweep", "Lakshadweep atolls", "coral", 72.63, 10.57, 60, 10, "coral atolls, lagoons"),
    ("kochi", "Kochi Port & Vembanad backwaters", "port", 76.25, 9.97, 12, 8, "port, backwater estuary, Chinese nets"),
    ("vizhinjam", "Vizhinjam Port", "port", 76.99, 8.37, 6, 6, "deep-water transhipment port"),
    ("kudankulam", "Kudankulam power station seawater intake", "intake", 77.71, 8.17, 5, 9, "nuclear plant cooling-water intake"),
    ("gulf_mannar", "Gulf of Mannar Marine National Park", "coral", 79.10, 9.15, 35, 10, "21 islands, coral reefs, dugong habitat"),
    ("tuticorin", "V.O. Chidambaranar (Tuticorin) Port", "port", 78.20, 8.76, 8, 6, "major port"),
    ("pichavaram", "Pichavaram mangroves", "mangrove", 79.78, 11.43, 8, 9, "mangrove forest"),
    ("kalpakkam", "Kalpakkam power station seawater intake", "intake", 80.17, 12.55, 5, 9, "nuclear plant cooling-water intake"),
    ("chennai", "Chennai & Kamarajar (Ennore) ports", "port", 80.30, 13.15, 12, 7, "major ports, Marina beach"),
    ("pulicat", "Pulicat Lake", "lagoon", 80.25, 13.60, 15, 9, "brackish lagoon, bird sanctuary"),
    ("coringa", "Coringa mangroves (Godavari delta)", "mangrove", 82.30, 16.80, 15, 9, "mangroves, wildlife sanctuary"),
    ("vizag", "Visakhapatnam Port", "port", 83.30, 17.68, 10, 6, "major port, naval base"),
    ("chilika", "Chilika Lake", "lagoon", 85.45, 19.70, 25, 10, "Ramsar lagoon, Irrawaddy dolphins"),
    ("paradip", "Paradip Port", "port", 86.68, 20.26, 8, 6, "major port, oil terminals"),
    ("gahirmatha", "Gahirmatha / Bhitarkanika", "turtle_mangrove", 87.00, 20.65, 20, 10, "olive ridley mass nesting, mangroves"),
    ("haldia", "Haldia / Kolkata port approaches", "port", 88.10, 21.95, 10, 6, "major port"),
    ("sundarbans", "Sundarbans", "mangrove", 88.90, 21.90, 40, 10, "largest mangrove forest, tiger habitat"),
    ("andaman_mgmnp", "Mahatma Gandhi Marine National Park", "coral", 92.60, 11.55, 20, 10, "coral reefs"),
    ("port_blair", "Port Blair", "port", 92.73, 11.67, 8, 6, "port"),
]

SITE_TYPE_LABEL = {
    "coral_mangrove": "Coral & mangroves", "coral": "Coral reef", "mangrove": "Mangroves", "lagoon": "Lagoon",
    "turtle_mangrove": "Turtle nesting & mangroves", "intake": "Seawater intake", "port": "Port", "fishing": "Fishing harbour",
    "tourism": "Tourism beaches",
}

# Indian ports used for the suspect-vessel "next port" estimate (lon, lat)
PORTS: list[tuple[str, float, float]] = [
    ("Deendayal (Kandla)", 70.22, 23.00), ("Mundra", 69.72, 22.74), ("Sikka", 69.83, 22.43), ("Pipavav", 71.53, 20.90),
    ("Mumbai", 72.84, 18.94), ("JNPT (Nhava Sheva)", 72.95, 18.95), ("Mormugao", 73.80, 15.41),
    ("New Mangalore", 74.80, 12.92), ("Kochi", 76.25, 9.97), ("Vizhinjam", 76.99, 8.37),
    ("V.O. Chidambaranar (Tuticorin)", 78.20, 8.76), ("Chennai", 80.30, 13.10), ("Kamarajar (Ennore)", 80.33, 13.26),
    ("Krishnapatnam", 80.13, 14.25), ("Visakhapatnam", 83.30, 17.68), ("Gangavaram", 83.23, 17.62),
    ("Paradip", 86.68, 20.26), ("Dhamra", 86.97, 20.80), ("Haldia", 88.10, 22.02), ("Port Blair", 92.73, 11.67),
]

# coastal places used to name a landing point
PLACES: list[tuple[str, float, float]] = [(s[1], s[3], s[4]) for s in SITES] + [(p[0], p[1], p[2]) for p in PORTS] + [
    ("Dwarka", 68.97, 22.24), ("Porbandar", 69.61, 21.64), ("Diu", 70.98, 20.71), ("Daman", 72.83, 20.41),
    ("Dahanu", 72.72, 19.97), ("Harihareshwar", 73.02, 17.99), ("Karwar", 74.12, 14.81), ("Udupi", 74.70, 13.34),
    ("Kasaragod", 74.99, 12.50), ("Kozhikode", 75.77, 11.25), ("Alappuzha", 76.33, 9.49), ("Kollam", 76.58, 8.88),
    ("Kanyakumari", 77.54, 8.08), ("Rameswaram", 79.31, 9.29), ("Nagapattinam", 79.84, 10.77),
    ("Puducherry", 79.83, 11.93), ("Nellore coast", 80.15, 14.45), ("Machilipatnam", 81.14, 16.17),
    ("Kakinada", 82.25, 16.93), ("Gopalpur", 84.91, 19.26), ("Puri", 85.83, 19.80), ("Digha", 87.51, 21.62),
    ("Jaffna (Sri Lanka)", 80.01, 9.66), ("Colombo (Sri Lanka)", 79.85, 6.93), ("Trincomalee (Sri Lanka)", 81.23, 8.57),
    ("Karachi (Pakistan)", 67.01, 24.86), ("Male (Maldives)", 73.51, 4.18),
]


def mrcc_for(lon: float, lat: float) -> dict:
    """Indian Coast Guard MRCC responsible for the area (simplified regional split)."""
    if lon >= 91.0:
        return {"name": "MRCC Port Blair", "region": "ICG Region (Andaman & Nicobar)"}
    if lon < 77.6 or (lat < 8.3 and lon < 78.5):
        return {"name": "MRCC Mumbai", "region": "ICG Region (West)"}
    return {"name": "MRCC Chennai", "region": "ICG Region (East)"}
