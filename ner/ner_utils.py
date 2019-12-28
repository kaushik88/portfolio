def get_entities(ents_obj):
    entities = {}
    for ent_obj in ents_obj:
        entity_name = ent_obj.label_
        if entity_name not in entities:
            entities[entity_name] = []
        entities[entity_name].append(ent_obj.text)
    return entities
