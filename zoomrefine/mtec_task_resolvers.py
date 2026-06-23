import re
from typing import Any, Dict, List, Optional


TASK_FAMILY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cross_shot_entity_count": {
        "resolver_class": "EntityBankCounter",
        "priority": "high",
        "required_evidence": ["entity_bank", "cross_shot_identity_merge", "role_filter", "duplicate_guard"],
    },
    "scene_group_attribute_count": {
        "resolver_class": "PanoramaAttributeCounter",
        "priority": "high",
        "required_evidence": ["best_wide_shot", "instance_attribute_table", "non_accumulation_guard"],
    },
    "temporal_event_count": {
        "resolver_class": "TemporalEventCounter",
        "priority": "high",
        "required_evidence": ["per_occurrence_timeline", "full_range_enumeration", "no_single_frame_count"],
    },
    "container_object_count": {
        "resolver_class": "ContainerObjectCounter",
        "priority": "high",
        "required_evidence": ["container_roi", "inside_outside_filter", "scoped_instance_count"],
    },
    "missing_set": {
        "resolver_class": "OptionConditionedVisibleSetResolver",
        "priority": "high",
        "required_evidence": ["option_visibility_table", "full_scope_seen_aggregation", "missing_candidates"],
    },
    "stateful_ocr": {
        "resolver_class": "StatefulOCRTracker",
        "priority": "high",
        "required_evidence": ["stable_ocr_roi", "multi_frame_ocr_votes", "state_machine_constraints"],
    },
    "ordinal_clip_action": {
        "resolver_class": "OrdinalClipActionResolver",
        "priority": "high",
        "required_evidence": ["logical_clip_boundaries", "target_ordinal_clip", "before_during_after_actions"],
    },
    "domain_intention": {
        "resolver_class": "DomainIntentResolver",
        "priority": "medium",
        "required_evidence": ["domain_ontology", "event_facts", "ranked_intents", "negative_evidence"],
    },
    "scene_conditioned_attribute": {
        "resolver_class": "SceneConditionedAttributeResolver",
        "priority": "medium",
        "required_evidence": ["scene_locator", "attribute_evidence", "scene_mismatch_guard"],
    },
    "generic_video_evidence": {
        "resolver_class": "GenericVideoEvidenceResolver",
        "priority": "normal",
        "required_evidence": ["global_timeline", "low_fps_context", "option_verification"],
    },
}


def parse_mcq_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for match in re.finditer(r"(?m)([A-F])\.\s*([^A-F\n][^\n]*)", str(question or "")):
        options[match.group(1).upper()] = " ".join(match.group(2).strip().split())
    return options


def route_question_family(question: str, options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    text = str(question or "").lower()
    options = options or parse_mcq_options(question)

    def has_any(terms: List[str]) -> bool:
        return any(term in text for term in terms)

    family = "generic_video_evidence"
    confidence = 0.55
    reasons: List[str] = []

    if has_any(["current score", "scoreboard", "score of", "比分", "当前比分", "计时器", "timer", "price", "license plate", "plate number", "displayed time"]):
        family = "stateful_ocr"
        confidence = 0.88
        reasons.append("state-like OCR keyword")
    elif has_any(["missing", "absent", "not shown", "not appear", "not visible", "not used", "not mentioned", "which color", "which of the following is not", "缺", "没有出现", "哪种没有", "哪个没有"]):
        family = "missing_set"
        confidence = 0.86
        reasons.append("absence or missing-set keyword")
    elif has_any(["first clip", "second clip", "third clip", "fourth clip", "last clip", "first segment", "second segment", "third segment", "第一个", "第二个", "第三个", "最后一段"]):
        family = "ordinal_clip_action"
        confidence = 0.86
        reasons.append("ordinal clip keyword")
    elif has_any(["intent", "intention", "purpose", "why", "goal", "ward", "grass", "bush", "brush", "moba", "enemy", "ally", "意图", "目的", "为什么"]):
        family = "domain_intention"
        confidence = 0.76
        reasons.append("intent or domain keyword")
    elif has_any(["how many", "number of", "count", "many", "几个", "多少"]):
        if is_action_repetition_count_question(question):
            # "how many jumps/rolls/spins/laps/sets/times ..." — counting a repeated
            # ACTION over time, not a static group in one frame. Route to temporal
            # event counting so occurrences are enumerated across the full timeline
            # instead of read off a single wide shot (which systematically undercounts).
            family = "temporal_event_count"
            confidence = 0.82
            reasons.append("count question over a repeated action/event across time")
        elif has_any(["box", "basket", "plate", "table", "bag", "container", "盒", "篮", "盘", "桌", "袋"]):
            family = "container_object_count"
            confidence = 0.84
            reasons.append("count question with container/surface scope")
        elif has_any(["men", "women", "male", "female", "boy", "girl", "stage", "on-stage", "wearing", "dressed", "hat", "red shirt", "男", "女", "舞台", "穿", "戴"]):
            family = "scene_group_attribute_count"
            confidence = 0.84
            reasons.append("count question with scene/group attribute scope")
        else:
            family = "cross_shot_entity_count"
            confidence = 0.78
            reasons.append("count question without static scene/container constraint")
    elif has_any(["wearing", "dressed", "on the train", "in the train", "in the scene", "at the scene", "在火车", "场景", "穿着", "戴着"]):
        family = "scene_conditioned_attribute"
        confidence = 0.68
        reasons.append("scene-conditioned attribute keyword")

    registry = TASK_FAMILY_REGISTRY[family]
    return {
        "task_family": family,
        "resolver_class": registry["resolver_class"],
        "priority": registry["priority"],
        "confidence": confidence,
        "route_reasons": reasons or ["default generic route"],
        "options_detected": sorted(options.keys()),
        "required_evidence": list(registry["required_evidence"]),
    }


def is_performing_cast_question(question: str) -> bool:
    """A scene-group count is a *performing cast* count when the people are
    performers/presenters who appear across multiple shots (often in close-up),
    rather than a static co-present group captured in one wide frame.

    These questions must be answered by deduped cross-shot unique-performer
    aggregation, NOT by a single best wide shot, because not all performers are
    simultaneously present in one frame.
    """
    text = str(question or "").lower()
    cast_terms = ["present", "presenting", "perform", "performing", "performer", "singer", "band", "host", "anchor", "presenter", "演", "主持", "表演"]
    stage_terms = ["stage", "on stage", "on-stage", "show", "concert", "舞台", "演出"]
    return any(term in text for term in cast_terms) and any(term in text for term in stage_terms)


# Repeated countable actions/events: "how many jumps/rolls/laps/sets/times ..." ask
# for the number of occurrences of an action over time, not a static group in one
# frame. The single-best-wide-shot counting method systematically under-counts these.
# Whole-word action nouns/verbs (matched with word boundaries to avoid substrings
# like "lap" in "laptop" or "spin" in "spinach").
_ACTION_REPETITION_WORDS = (
    "jump", "jumps", "roll", "rolls", "spin", "spins", "rotation", "rotations",
    "somersault", "somersaults", "cartwheel", "cartwheels", "lap", "laps", "rep",
    "reps", "push-up", "push-ups", "pushup", "pushups", "sit-up", "sit-ups",
    "squat", "squats", "kick", "kicks", "punch", "punches", "clap", "claps",
    "bounce", "bounces", "dribble", "dribbles", "swing", "swings", "round", "rounds",
    "set", "sets",
)
# Multi-word / phrase cues (safe as substrings) and CJK markers.
_ACTION_REPETITION_PHRASES = ("how many times", "times does", "times did", "次", "圈", "组")


def is_action_repetition_count_question(question: str) -> bool:
    """True for counting questions about a repeated action/event over time
    (jumps, rolls, laps, sets, 'how many times', ...), which must be counted by
    enumerating occurrences across the full timeline rather than from one frame."""
    text = str(question or "").lower()
    if not any(term in text for term in ("how many", "number of", "count", "many", "几个", "多少", "几次")):
        return False
    if any(p in text for p in _ACTION_REPETITION_PHRASES):
        return True
    words = set(re.findall(r"[a-z][a-z\-]*", text))
    return any(w in words for w in _ACTION_REPETITION_WORDS)


def build_task_specific_resolver_guidance(question: str) -> Dict[str, Any]:
    route = route_question_family(question)
    family = route["task_family"]
    template = _family_evidence_template(family, question)
    scope_guard = _question_scope_guard(question)
    count_mode = (
        "performing_cast_unique_across_shots"
        if family == "scene_group_attribute_count" and is_performing_cast_question(question)
        else None
    )
    guidance = {
        "version": "task_family_resolver_registry_v1",
        "task_family": family,
        "resolver_type": family,
        "resolver_class": route["resolver_class"],
        "priority": route["priority"],
        "route_confidence": route["confidence"],
        "route_reasons": route["route_reasons"],
        "required_evidence": route["required_evidence"],
        "confidence_gate": {
            "hard_result_min_confidence": 0.75,
            "joint_reasoning_min_confidence": 0.50,
            "below_0_50": "fallback_to_generic_video_evidence",
        },
        "question_scope_guard": scope_guard,
        "evidence_template": template,
        "rules": _family_rules(family, question) + scope_guard.get("rules", []),
        "fallback_policy": (
            "Resolver outputs evidence only. The final verifier still maps evidence to options. "
            "If resolver evidence is incomplete, conflicting, or below confidence gate, re-check the full-video anchor and generic evidence."
        ),
    }
    if count_mode:
        guidance["count_mode"] = count_mode
    return guidance


def _family_evidence_template(family: str, question: Optional[str] = None) -> Dict[str, Any]:
    if family == "temporal_event_count":
        return {
            "target_action": "question_action",
            "count_method": "enumerate_each_occurrence_across_full_timeline_then_sum",
            "scan_scope": "full_question_time_range",
            "occurrences": [
                {"index": 1, "time_sec": None, "actor": "", "evidence": "", "confidence": 0.0}
            ],
            "count_value": None,
            "undercount_check": "scanned every segment of the relevant range; fast/repeated reps not skipped",
            "count_confidence": 0.0,
        }
    if family == "scene_group_attribute_count" and is_performing_cast_question(question or ""):
        return {
            "scene": "stage_performance",
            "count_method": "unique_cast_across_shots_dedup",
            "include_closeup_performers": True,
            "unique_performers": [
                {
                    "performer_id": "P1",
                    "first_seen": "",
                    "shot_types": [],
                    "gender": "",
                    "role": "",
                    "dedup_key": "",
                    "include_in_count": True,
                    "include_reason": "",
                    "confidence": 0.0,
                }
            ],
            "excluded": ["audience", "crew", "host_if_not_performing"],
            "total_people": None,
            "attribute_breakdown": {"men": None, "women": None},
            "count_confidence": 0.0,
        }
    if family == "cross_shot_entity_count":
        return {
            "target_entity": "question_target",
            "role_filter": ["foreground", "participant", "interacts_with_task"],
            "exclude": ["audience", "background", "host", "referee_if_not_target"],
            "entities": [
                {
                    "global_entity_id": "E1",
                    "local_tracks": [],
                    "role": "",
                    "appearance": "",
                    "include_in_count": False,
                    "include_reason": "",
                    "confidence": 0.0,
                }
            ],
            "count_value": None,
            "count_confidence": 0.0,
        }
    if family == "scene_group_attribute_count":
        return {
            "scene": "question_scene",
            "count_method": "best_wide_shot_not_sum_across_shots",
            "selected_frame_or_range": "",
            "total_people": None,
            "attribute_breakdown": {},
            "ignored_closeups": [],
        }
    if family == "container_object_count":
        return {
            "container": "question_container",
            "scope": "beginning_or_question_scope",
            "container_roi": [],
            "visible_items_inside_container": [],
            "excluded_outside_items": [],
            "count_value": None,
        }
    if family == "missing_set":
        return {
            "option_visibility": {},
            "visible_set": [],
            "missing_candidates": [],
            "rule": "absence_by_option_wise_full_scope_aggregation",
        }
    if family == "stateful_ocr":
        return {
            "target_state": "score_or_displayed_text",
            "stable_roi": {},
            "ocr_sequence": [],
            "selected_state": "",
            "state_confidence": 0.0,
            "state_machine_notes": [],
        }
    if family == "ordinal_clip_action":
        return {
            "logical_clips": [],
            "target_clip_index": None,
            "target_logical_clip_range": "",
            "before_during_after": {},
            "action_hypotheses": [],
            "clip_uncertainties": [],
        }
    if family == "domain_intention":
        return {
            "domain": "auto_detected",
            "ontology_terms": [],
            "event_facts": [],
            "ranked_intents": [],
            "negative_evidence": [],
        }
    if family == "scene_conditioned_attribute":
        return {
            "target_scene": "question_scene",
            "scene_locator": {},
            "target_subject": "",
            "attribute_evidence": [],
            "scene_mismatch_evidence": [],
        }
    return {"facts": [], "conflicts": [], "fallback_needed": False}


def _family_rules(family: str, question: Optional[str] = None) -> List[str]:
    common = [
        "Do not output candidate_answer, preliminary_answer, best_option, final_answer, or a final option letter.",
        "Every non-empty evidence item must include timestamp/time range and anchor_link when available.",
        "Resolver output is structured evidence only; final option mapping happens later.",
    ]
    if family == "scene_group_attribute_count" and is_performing_cast_question(question or ""):
        return common + [
            "This is a performing/presenting cast count: count every distinct performer who appears on the stage across the WHOLE performance, not only those visible in a single wide frame.",
            "A performer seen only in close-up (for example a lead singer) still counts once; do NOT drop or ignore close-up performers.",
            "Build a unique-performer bank and dedupe the same person across shots using face, outfit, hair, and role; record uncertain merges.",
            "Include lead performers, presenters, and backup performers; exclude only the audience and crew.",
            "After building the deduped cast, map it to men/women, then verify every multiple-choice option against the cast bank instead of stopping at the first plausible wide-shot count.",
            "The number of people in any single frame is a lower bound, not the answer; the deduped cross-shot cast total is usually higher.",
        ]
    family_rules = {
        "temporal_event_count": [
            "This counts a repeated ACTION/EVENT over time (e.g. jumps, rolls, laps, sets, 'how many times'), NOT a static group in one frame.",
            "Enumerate EACH occurrence with its own timestamp across the full relevant time range, then the count is the number of occurrences (sum), not what is visible in any single frame.",
            "Scan the entire relevant span including fast or back-to-back repetitions; compressed/low-FPS frames make it easy to SKIP occurrences, so the true count is usually HIGHER than a quick glance suggests.",
            "If unsure between two adjacent option counts, prefer the higher one unless you can positively account for every occurrence and rule the higher count out.",
            "Use the low-FPS evidence video and the global timeline together to catch occurrences between sampled frames.",
        ],
        "cross_shot_entity_count": [
            "Build an entity bank across the relevant timeline; never answer from one frame or one crop.",
            "Merge the same entity across shots using appearance, role, position, and repeated participation; record uncertain merges.",
            "Exclude background people, audience, hosts, and non-participants unless the question target includes them.",
        ],
        "scene_group_attribute_count": [
            "Use the best wide/panorama shot as the primary count; do not sum close-ups or repeated shots.",
            "For gender/clothing/attribute questions, create one row per visible person in the selected scene frame.",
            "Use additional frames only to verify unclear attributes, not to add duplicate people.",
            "For multiple-choice count options, verify each option independently against the wide shot; do not stop at the first plausible count.",
            "For presenting/on-stage questions, include lead performers, presenters, and backup performers visible on the stage; do not count only the central dancers.",
            "If the model's free count does not match any option exactly, re-check all stage-edge people before selecting an option.",
        ],
        "container_object_count": [
            "First locate the container or surface ROI, then count only instances inside it.",
            "For beginning/start scopes, later ASR or later frames cannot override opening visual count.",
            "Record excluded outside-container objects separately.",
            "When the question says displayed at the beginning, do not use later reveal shots, later pack shots, or product-list narration as primary evidence.",
            "For multiple-choice count options, verify each option count against the scoped container ROI instead of trusting transcript numbers.",
        ],
        "missing_set": [
            "Check every option independently across the valid time scope.",
            "Visible once is enough for seen=true; unseen in one frame is not enough for missing.",
            "Missing candidates are options with no positive evidence after full-scope aggregation.",
        ],
        "stateful_ocr": [
            "Use direct high-resolution OCR crops and adjacent frames; do not infer text/score from context.",
            "Vote across multiple frames and prefer stable states over isolated low-confidence OCR jumps.",
            "For scores, a change needs visual scoring-event support or repeated stable OCR.",
        ],
        "ordinal_clip_action": [
            "Logical clip index is not the same as PySceneDetect atomic shot index.",
            "Group adjacent shots/replays/context belonging to one event before selecting the ordinal clip.",
            "Describe before/during/after evidence for the target logical clip.",
        ],
        "domain_intention": [
            "Detect the domain first and map facts to domain ontology terms.",
            "Rank intent hypotheses using positive and negative evidence.",
            "Do not infer intent from a single generic action when a domain-specific alternative explains it better.",
        ],
        "scene_conditioned_attribute": [
            "Locate the target scene before judging attributes.",
            "Treat evidence from other scenes as scene_mismatch unless explicitly linked to the target subject.",
            "Use visual attributes from the located scene over transcript guesses.",
        ],
        "generic_video_evidence": [
            "Use global timeline, local crops, transcript, and options jointly.",
            "Record uncertainty instead of forcing unsupported evidence.",
        ],
    }
    return common + family_rules.get(family, family_rules["generic_video_evidence"])


def _question_scope_guard(question: str) -> Dict[str, Any]:
    text = str(question or "").lower()
    if any(term in text for term in ["beginning", "at the start", "start of", "opening", "initially", "displayed at the beginning"]):
        return {
            "scope_type": "beginning_locked",
            "allowed_scope": "opening visual segment only",
            "forbidden_scope": "later reveal shots, later flat-lay/pack shots, later transcript/ASR claims, later product summaries",
            "rules": [
                "The question explicitly locks the evidence to the beginning/opening segment. The COUNT must come from the opening segment only.",
                "Forbidden: deriving the count from any later reveal shot, later flat-lay, end-state, or product-summary narration, even if those shots are clearer or easier to count.",
                "If a later spoken phrase states a product/item number (for example 'eight full-size products') during a later reveal, it is end-scope narration and must NOT be used as the beginning count.",
                "If the beginning box/container is partly occluded, count the visible items in the opening segment and record uncertainty; do not substitute a later, clearer reveal count.",
            ],
        }
    return {"scope_type": "full_or_question_scope", "rules": []}
