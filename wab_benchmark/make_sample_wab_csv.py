# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import re
import pandas as pd

data = {
    'object': {
        'turn on the stove and put the moka pot on it': 'Make coffee using the moka pot on the stove.',
        'put the black bowl in the bottom drawer of the cabinet and close it': 'Put the black bowl into the bottom drawer of the cabinet and close the drawer.',
        'put the yellow and white mug in the microwave and close it': 'Put the mug inside the microwave and close the door.',
        'put both moka pots on the stove': 'Prepare hot coffee by using the moka pots available in the kitchen.',
        'LIVING ROOM SCENE1 put both the alphabet soup and the cream cheese box in the basket': 'Put the alphabet soup and cream cheese box into the basket in the living room.',
        'LIVING ROOM SCENE2 put both the alphabet soup and the tomato sauce in the basket': 'Put the alphabet soup and tomato sauce into the basket.',
        'LIVING ROOM SCENE2 put both the cream cheese box and the butter in the basket': 'Put the cream cheese and butter into the basket in the living room.',
        'LIVING ROOM SCENE5 ut the white mug on the left plate and put the yellow and white mug on the right plate': 'Place the white mug on the left plate and the yellow-and-white mug on the right plate in the living room.',
        'LIVING ROOM SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate': 'Place the white mug on the plate and position the chocolate pudding to the right of it.',
        'STUDY SCENE1 pick up the book and place it in the back compartment of the caddy': 'Put the book into the back section of the caddy.'
    },
    '10': {
        'pick up the alphabet soup and place it in the basket': 'Put the alphabet soup can into the shopping basket.',
        'pick up the bbq sauce and place it in the basket': 'Put the BBQ sauce in the basket.',
        'pick up the butter and place it in the basket': 'Put the butter in the basket.',
        'pick up the chocolate pudding and place it in the basket': 'Put the chocolate pudding into the basket.',
        'pick up the cream cheese and place it in the basket': 'Put the cream cheese in the basket.',
        'pick up the ketchup and place it in the basket': 'Put the ketchup bottle into the basket.',
        'pick up the milk and place it in the basket': 'Put the milk carton in the shopping basket.',
        'pick up the orange juice and place it in the basket': 'Put the orange juice in the basket.',
        'pick up the salad dressing and place it in the basket': 'Put the salad dressing in the basket.',
        'pick up the tomato sauce and place it in the basket': 'Put the tomato sauce in the basket.'
    }
}

def norm(s): 
    return re.sub(r"\s+", " ", s.strip().lower())

def derive_gold(instr_raw: str):
    t = norm(instr_raw)

    if "turn on the stove" in t and ("put the moka pot" in t or "moka pot on it" in t or "moka" in t):
        return ["turn_on(stove)", "place_on(moka_pot,stove)"]

    if "drawer" in t or "cabinet" in t:
        tgt = "bottom_drawer" if "bottom drawer" in t else "drawer"
        # open/put_in/close
        obj = "black_bowl" if "black bowl" in t else "item"
        return [f"open({tgt})", f"put_in({obj},{tgt})", f"close({tgt})"]

    if "microwave" in t:
        obj = "mug" if "mug" in t else "item"
        return [f"open(microwave)", f"put_in({obj},microwave)", f"close(microwave)"]

    if "put both" in t and "stove" in t:
        return ["place_on(moka_pot,stove)", "place_on(moka_pot,stove)"]

    if "basket" in t:
        # 여러 개면 2회 put_in
        many = any(k in t for k in ["both", "and"])
        if many:
            return ["put_in(item,basket)", "put_in(item,basket)"]
        return ["put_in(item,basket)"]

    if "plate" in t and ("left" in t or "right" in t):
        return ["place_on(white_mug,plate)", "place_on(yellow_white_mug,plate)"]

    if "plate" in t and "right of the plate" in t:
        return ["place_on(white_mug,plate)", "place_on(chocolate_pudding,desk_right_of_plate)"]

    if "back compartment" in t or "back section" in t:
        return ["pick(book)", "put_in(book,caddy_back)"]

    if "plate" in t:
        return ["pick(item)", "place_on(item,plate)"]
    if "stove" in t:
        return ["place_on(item,stove)"]
    return ["pick(item)", "place_on(item,surface)"]

rows = []
for group, mapping in data.items():
    for raw_instr, transformed in mapping.items():
        instruction_en = transformed 
        gold = derive_gold(raw_instr) 
        rows.append({
            "group": group,
            "raw_instruction": raw_instr,
            "instruction_en": instruction_en,
            "gold_plan": "; ".join(gold)
        })

df = pd.DataFrame(rows)
df.to_csv("sample_wab.csv", index=False)
print("Saved: sample_wab.csv")
print(df.head())
