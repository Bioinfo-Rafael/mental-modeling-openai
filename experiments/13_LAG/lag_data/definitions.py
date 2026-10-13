"""Dataset field order, units, action IDs, and the supplied CSV schema."""
from decimal import Decimal

DT = Decimal("0.2")
AIRCRAFT = ("A0100", "A0200", "B0100", "B0200")
TRANSFORM = ("lon_deg", "lat_deg", "alt_m", "roll_deg", "pitch_deg", "yaw_deg")
VELOCITY = ("tas_mps", "vn_mps", "ve_mps", "vu_mps")
ACTION_LABELS = {
    "altitude_action": ("up", "hold", "down"),
    "heading_action": ("left30", "left15", "hold", "right15", "right30"),
    "speed_action": ("up", "hold", "down"),
    "shoot_action": ("no", "yes"),
}
PROBABILITIES = tuple(f"p_{action.removesuffix('_action')}_{label}"
                      for action, labels in ACTION_LABELS.items() for label in labels)
CSV_FIELDS = ("step", "time_s", "uid", "color", "controller", "alive",
              *TRANSFORM, *VELOCITY,
              *(field for action, labels in ACTION_LABELS.items()
                for field in (action, *(f"p_{action.removesuffix('_action')}_{x}" for x in labels))),
              "missile_launched")
CSV_RENAMES = {"uid": "object_id"}
COLUMNS = (
    "episode_id", "step", "time_s", "object_id", "entity_type", "team", "color",
    "name", "acmi_type", "launcher_id", "controller", "alive",
    *TRANSFORM, *VELOCITY, "health", "missiles_remaining",
    *ACTION_LABELS, *(f"{x}_id" for x in ACTION_LABELS), *PROBABILITIES,
    "missile_launched", "acmi_event", "radius_m", "has_csv_record", "has_acmi_record",
    "state_valid", "state_source", "action_status", "csv_row_number", "acmi_line_number",
    "acmi_time_s", *(f"acmi_{x}" for x in TRANSFORM), "acmi_tas_mps",
    "acmi_properties_json",
)
INTEGER_COLUMNS = {"step", "alive", "missiles_remaining", "has_csv_record", "has_acmi_record",
                   "state_valid", "csv_row_number", "acmi_line_number",
                   *(f"{x}_id" for x in ACTION_LABELS)}
FLOAT_COLUMNS = {"time_s", *TRANSFORM, *VELOCITY, "health", "radius_m", *PROBABILITIES,
                 "acmi_time_s", *(f"acmi_{x}" for x in TRANSFORM), "acmi_tas_mps"}


ACTIONS = tuple(ACTION_LABELS)
STATE_FEATURES = TRANSFORM + VELOCITY
STATE_UNITS = {
    "lon_deg": "degrees", "lat_deg": "degrees", "alt_m": "m",
    "roll_deg": "degrees", "pitch_deg": "degrees", "yaw_deg": "degrees",
    "tas_mps": "m/s", "vn_mps": "m/s", "ve_mps": "m/s", "vu_mps": "m/s",
}
