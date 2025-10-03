"""
In machine learning, an ontology is a structured way to represent knowledge about a domain.
It defines concepts, relationships, and categories in a formalized way so that machines can "understand" the context
and meaning of data.

Think of it as a map of concepts with rules about how they relate to each other.

Healthcare ontology:

Patient --has--> Disease
Disease --has symptom--> Symptom
Medication --treats--> Disease
"""

from owlready2 import *

# Create a new ontology
onto = get_ontology("http://example.org/healthcare.owl")

with onto:
    # Define classes (concepts)
    class Patient(Thing): pass
    class Disease(Thing): pass
    class Symptom(Thing): pass
    class Medication(Thing): pass

    # Define relationships
    class has_disease(Patient >> Disease): pass
    class has_symptom(Disease >> Symptom): pass
    class treats(Medication >> Disease): pass

# Create instances (entities)
flu = Disease("Flu")
fever = Symptom("Fever")
cough = Symptom("Cough")
aspirin = Medication("Aspirin")

# Define relations
flu.has_symptom = [fever, cough]
aspirin.treats = [flu]

# Create a patient
john = Patient("John")
john.has_disease = [flu]

# Query ontology
print(f"Patient {john.name} has diseases: {[d.name for d in john.has_disease]}")
for disease in john.has_disease:
    print(f"Disease {disease.name} has symptoms: {[s.name for s in disease.has_symptom]}")


# Check which medications can treat John's diseases
for med in onto.Medication.instances():
    for disease in john.has_disease:
        if disease in med.treats:
            print(f"{med.name} can treat {disease.name}")
