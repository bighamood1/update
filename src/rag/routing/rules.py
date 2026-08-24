"""Rule tables for the deterministic query router (no LLM involved).

Faculty aliases are normalized (Arabic diacritics/hamza-stripped) so variants
like ``الاعمال``/``الأعمال`` both match. Aliases are matched as substrings of
the normalized query; when several faculties match, the LONGEST alias wins so
``العلوم الصحية التطبيقية`` is not mis-read as the generic ``science``.
"""

from __future__ import annotations

# intent -> weighted keyword list. Higher weight = stronger signal.
INTENT_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "COMPARISON": [
        ("compare", 2.0), ("comparison", 2.0), ("difference", 2.0),
        ("vs", 1.5), ("versus", 1.5), ("between", 1.0),
        ("مقارنة", 2.0), ("الفرق", 2.0), ("أيهما", 2.0), ("بين", 1.0),
    ],
    "LOCATION": [
        ("located", 1.5), ("location", 1.5), ("where", 1.2), ("address", 1.2),
        ("أين", 1.5), ("فين", 1.5), ("تقع", 1.5),
        ("العنوان", 1.5), ("عنوان", 1.4), ("موقع", 1.2), ("مكان", 1.2),
        ("موجودة", 1.0), ("موجوده", 1.0),
    ],
    "PERSON": [
        ("president", 2.0), ("dean", 2.0), ("minister", 2.0), ("who", 1.0),
        ("رئيس", 2.0), ("عميد", 2.0), ("من هو", 1.5), ("من يكون", 1.5),
        ("مين", 1.5), ("مين هو", 1.5),
    ],
    "ADMINISTRATION": [
        ("board of trustees", 2.0), ("council", 1.5), ("administration", 1.5),
        ("management", 1.2), ("إدارة", 1.5), ("مجلس الأمناء", 2.0),
        ("مجلس الجامعة", 2.0), ("الإدارة", 1.5),
    ],
    "ADMISSION": [
        ("admission", 2.0), ("apply", 1.2), ("application", 1.2),
        ("enroll", 1.5), ("requirements", 1.0), ("acceptance", 1.2),
        ("secondary school", 1.5), ("قبول", 2.0), ("التحاق", 2.0),
        ("شروط", 2.0), ("ثانوية", 1.5), ("تقديم", 1.2), ("تسجيل", 1.2),
    ],
    "REGULATION": [
        ("regulation", 2.0), ("rules", 1.5), ("transfer", 1.5), ("policy", 1.2),
        ("law", 1.0), ("code of ethics", 2.0), ("قواعد", 2.0), ("التحويل", 2.0),
        ("لوائح", 2.0), ("سياسة", 1.2), ("قانون", 1.0),
    ],
    "TUITION": [
        ("tuition", 2.0), ("fees", 1.5), ("fee", 1.5),
        ("how much", 2.0), ("cost", 1.5), ("price", 1.2),
        ("expenses", 1.2), ("مصروفات", 2.0), ("مصاريف", 2.0),
        ("رسوم", 2.0), ("تكاليف", 1.5), ("تكلفة", 1.5),
        ("كام", 2.0), ("بكام", 2.0),
    ],
    "SCHOLARSHIP": [
        ("scholarship", 2.0), ("scholarships", 2.0), ("grant", 1.2),
        ("financial aid", 1.5), ("منحة", 2.0), ("منح", 2.0),
        ("المنح", 2.0), ("الدعم الاجتماعي", 1.5), ("خصم", 1.0),
    ],
    "FACILITY": [
        ("facility", 1.5), ("facilities", 1.5), ("library", 1.5), ("lab", 1.0),
        ("مرافق", 1.5), ("مكتبة", 1.5), ("معامل", 1.0),
    ],
    "NEWS": [
        ("news", 1.5), ("latest", 1.0), ("event", 1.0), ("أخبار", 1.5),
        ("فعاليات", 1.0), ("الأحداث", 1.2),
    ],
    "CONTACT": [
        ("contact", 2.0), ("phone", 1.5), ("email", 1.5), ("reach", 1.2),
        ("اتصال", 2.0), ("تواصل", 2.0), ("هاتف", 1.5), ("بريد", 1.2),
    ],
    "FAQ": [
        ("faq", 2.0), ("frequently asked", 2.0), ("study system", 1.5),
        ("كيف", 1.0), ("نظام", 0.8), ("الأسئلة الشائعة", 2.0),
    ],
    "FACULTY": [
        ("faculty", 1.5), ("faculties", 1.5), ("college", 1.0),
        ("كلية", 1.5), ("كليات", 1.5),
    ],
    "PROGRAM": [
        ("program", 1.5), ("programs", 1.5), ("course", 1.0), ("degree", 1.0),
        ("department", 1.2), ("departments", 1.2), ("major", 1.0), ("majors", 1.0),
        ("برنامج", 1.5), ("برامج", 1.5), ("مقرر", 1.0), ("قسم", 1.2),
        ("أقسام", 1.2), ("اقسام", 1.2), ("تخصص", 1.0),
    ],
    "LIST": [
        ("list", 1.5), ("list all", 2.0), ("all the", 1.0), ("all of", 1.0),
        ("how many", 1.5), ("name the", 1.5), ("enumerate", 2.0),
        ("available", 0.8), ("قائمة", 2.0), ("اذكر", 2.0), ("جميع", 1.5),
        ("عدد", 1.5), ("ما هي", 0.8), ("كل", 0.5),
    ],
}

# intent -> (category, concrete content_type values for Chroma filtering)
CATEGORY_MAP: dict[str, tuple[str, list[str]]] = {
    "ADMISSION": ("admissions", ["admission", "tuition", "faq", "about", "home"]),
    "TUITION": ("tuition", ["tuition", "admission", "faq", "about", "home"]),
    "SCHOLARSHIP": ("scholarships", ["scholarship", "about", "home", "news"]),
    "FACULTY": ("faculties", ["faculty", "program", "about", "home"]),
    "PROGRAM": ("programs", ["program", "faculty", "about", "home"]),
    "LIST": ("general", ["faculty", "program", "about", "home", "administration"]),
    "LOCATION": ("location", ["about", "home", "faculty", "contact"]),
    "PERSON": ("people", ["president", "administration", "about", "home"]),
    "ADMINISTRATION": ("administration", ["administration", "president", "about", "home"]),
    "REGULATION": ("regulations", ["regulation", "policy", "guide", "about"]),
    "FACILITY": ("facilities", ["facility", "about", "home"]),
    "NEWS": ("news", ["news", "event", "home"]),
    "CONTACT": ("contact", ["contact", "about", "home"]),
    "FAQ": ("faq", ["faq", "about", "home", "admission"]),
    "COMPARISON": ("general", []),  # multi-category by design
}

# intent -> content types that are "primary" for the intent. A confident route
# boosts chunks of these types during fusion so e.g. an ADMISSION question
# surfaces the actual admission page rather than generic news/home pages.
PRIORITY_TYPES: dict[str, list[str]] = {
    "ADMISSION": ["admission"],
    "TUITION": ["tuition"],
    "SCHOLARSHIP": ["scholarship"],
    "FACULTY": ["faculty", "program"],
    "PROGRAM": ["program", "faculty"],
    "LIST": ["faculty", "program"],
    "LOCATION": ["about"],
    "PERSON": ["president", "administration"],
    "ADMINISTRATION": ["administration"],
    "REGULATION": ["regulation", "policy"],
    "FACILITY": ["facility"],
    "NEWS": ["news"],
    "CONTACT": ["contact"],
    "FAQ": ["faq"],
}

# canonical faculty key -> alias list (normalized lowercase)
FACULTY_ALIASES: dict[str, list[str]] = {
    "business": ["business", "الاعمال", "الأعمال", "ادارة الاعمال", "إدارة الأعمال"],
    "law": ["law", "القانون"],
    "engineering": ["engineering", "الهندسة", "هندسة"],
    "computer-science-and-engineering": [
        "computer science", "computer science & engineering",
        "computer science and engineering", "cse", "علوم وهندسة الحاسب",
        "كلية الحاسب", "كلية الحاسبات", "الحاسب", "علوم الحاسب", "هندسة الحاسب",
    ],
    "textile-science-and-engineering": [
        "textile", "علوم وهندسة المنسوجات", "المنسوجات",
    ],
    "science": ["science", "العلوم"],
    "medicine": ["medicine", "الطب", "طب"],
    "dentistry": ["dentistry", "طب الاسنان", "طب الأسنان", "الاسنان"],
    "pharmacy": ["pharmacy", "الصيدلة"],
    "social-and-human-sciences": [
        "social", "العلوم الاجتماعية", "الإنسانية", "الانسانية",
    ],
    "applied-health-sciences": [
        "applied health", "العلوم الصحية", "الصحية التطبيقية",
    ],
    "nursing": ["nursing", "التمريض"],
    "graduate-studies": ["graduate studies", "الدراسات العليا", "العليا"],
    "mass-media-and-communication": [
        "mass media", "الإعلام", "الاعلام", "الاتصال",
    ],
    "physical-therapy": ["physical therapy", "العلاج الطبيعي", "الطبيعي"],
}

# Strong markers that raise confidence (question openings).
_STRONG_MARKERS = (
    "what", "who", "where", "when", "how", "which", "why", "list", "name",
    "ما", "من", "أين", "متى", "كيف", "اذكر", "عدد", "فين",
)
