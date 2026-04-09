from classifier import get_classifier

clf = get_classifier()

tests = [
    "2018 flood tragedy turns into political wrangle in Kerala on poll eve",
    "Floods hit Assam, thousands evacuated",
    "Debate on earthquake preparedness in parliament",
    "Forest fire spreads rapidly in Himachal Pradesh",
    "Earthquake jolts Nepal border region",
    "Cyclone alert issued as heavy winds approach Odisha",
    "Remembering the 2013 Kedarnath disaster"
]

for t in tests:
    label, conf = clf.predict(t)
    print("\n" + "=" * 80)
    print("TEXT:", t)
    print("PREDICTION:", label)
    print("CONFIDENCE:", conf)