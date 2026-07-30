"""The five personas behind demo_corpus/ — a synthetic social circle whose
messages, photos, and facts are entirely invented. No real person is
depicted or referenced anywhere in this module or the corpus it drives.

Base full names were seeded from `Faker(seed=75)`:

    Jeffrey Williams | Joshua Vega | Cory Davis | Jill Smith | Kathryn Henry

...then hand-noised per persona, same spirit as the plan's own worked example
("Sarah Kovacs" -> whatsapp "Sarah", instagram "sarah.kovacs", sms "Sarah K
\U0001f483"): "Jill Smith" became "Sarah Mendes" (Jeffrey's WhatsApp 1:1
partner — the corpus needs a "sarah" thread), and "Kathryn Henry" became
"Kathryn Petrović" (surname swapped in to exercise the Instagram mojibake
round-trip on a real diacritic, and to ground her Croatia backstory in the
conversation data). Jeffrey Williams is the protagonist: the export's device
owner, present in every thread.

Platform variants are deliberately noisy on purpose — different apps show
different names for the same person, and entity resolution (a later task)
has to reconcile "Kathryn Petrović" (Instagram), "Kat" (WhatsApp), and
"Kat P \U0001f4f7" (SMS contact name) back into one entity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    canonical: str
    whatsapp_name: str
    instagram_handle: str  # what appears as sender_name in the IG export JSON
    sms_contact_name: str | None  # None for the device owner (sms sender is "me")
    phone: str
    face_id: str | None = None


PERSONAS: list[Persona] = [
    Persona(
        canonical="Jeffrey Williams",
        whatsapp_name="Jeffrey Williams",
        instagram_handle="Jeffrey Williams",
        sms_contact_name=None,  # device owner — his own sms messages are sender="me"
        phone="+18872515144",
        face_id="face_01",
    ),
    Persona(
        canonical="Sarah Mendes",
        whatsapp_name="Sarah Mendes",
        instagram_handle="sarah.mendes",
        sms_contact_name="Sarah M ⭐",
        phone="+17886704852",
        face_id="face_02",
    ),
    Persona(
        canonical="Kathryn Petrović",
        whatsapp_name="Kathryn Petrović",
        instagram_handle="Kathryn Petrović",  # her real IG display name, diacritic and all
        sms_contact_name="Kat P \U0001f4f7",
        phone="+16648588751",
        face_id="face_03",
    ),
    Persona(
        canonical="Joshua Vega",
        whatsapp_name="Joshua Vega",
        instagram_handle="joshvega_",
        sms_contact_name="Josh V",
        phone="+13322324792",
        face_id="face_04",
    ),
    Persona(
        canonical="Cory Davis",
        whatsapp_name="Cory Davis",
        instagram_handle="corydavis.photo",
        sms_contact_name="Cory D",
        phone="+12146581744",
        face_id="face_05",
    ),
]

BY_CANONICAL: dict[str, Persona] = {p.canonical: p for p in PERSONAS}
