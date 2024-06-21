from sys import stderr, argv
from typing import Optional


class Hashable:
    """
    An object whose hash and equality are based solely on a `name` variable.
    Used when placing students/hospitals in sets.
    """

    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Student(Hashable):
    """
    A resident in the Gale-Shapley algorithm.
    """

    def __init__(self, name: str):
        self.name = name
        self.hospital_ranking: list["Hospital"] = []
        self.match: Optional["Hospital"] = None

    def apply_to_top_hospital(self):
        """
        Apply to the top-ranked hospital, then remove it from the list.
        """

        self.hospital_ranking.pop(0).add_applicant(self)


class Hospital(Hashable):
    """
    A hospital in the Gale-Shapley algorithm.
    """

    def __init__(self, name: str):
        self.name = name
        self.student_ranking: list["Student"] = []
        self.applicants: list["Student"] = []
        self.match: Optional["Student"] = None

    def add_applicant(self, student: "Student"):
        """
        Accept an application from a student.
        """
        self.applicants.append(student)

    def sort_applicants(self):
        """
        Sort the list of applicants by the hospital's own preferences.
        """

        self.applicants = [
            student
            for student in self.student_ranking
            if student in set(self.applicants)
        ]


def main():
    if len(argv) != 2:
        stderr.write("Usage: python matching.py <filename>")
        exit(1)

    students, hospitals = load(argv[1])

    gale_shapley_matching(students, hospitals)

    print_matches(students)


def load(filename: str) -> tuple[list[Student], list[Hospital]]:
    """
    Load input data from a file.
    """

    with open(filename, "r", encoding="ascii") as f:
        raw_data = [line.strip() for line in f.readlines()]

    try:
        n = int(raw_data[0])
    except TypeError:
        stderr.write(f"Line 1 of input file ({raw_data[0]}) is not an integer!")
        exit(1)

    if len(raw_data) != n * 2 + 1:
        stderr.write(f"Expected {n*2 + 1} lines, got {len(raw_data)} lines.")
        exit(1)

    students_raw = raw_data[1 : n + 1]
    hospitals_raw = raw_data[n + 1 :]

    # Initialize all Student and Hospital objects
    students = [Student(student_raw.split(" ")[0]) for student_raw in students_raw]
    hospitals = [Hospital(hospital_raw.split(" ")[0]) for hospital_raw in hospitals_raw]

    # Fill each entity's ranked list with these objects
    for student, student_raw in zip(students, students_raw):
        hospital_ranking = student_raw.split(" ")[1:]
        for hospital_name in hospital_ranking:
            student.hospital_ranking.append(
                [hospital for hospital in hospitals if hospital.name == hospital_name][
                    0
                ]
            )
    for hospital, hospital_raw in zip(hospitals, hospitals_raw):
        student_ranking = hospital_raw.split(" ")[1:]
        for student_name in student_ranking:
            hospital.student_ranking.append(
                [student for student in students if student.name == student_name][0]
            )

    return students, hospitals


def gale_shapley_matching(students: list[Student], hospitals: list[Hospital]):
    """
    Gale-Shapley stable matching algorithm.
    """

    # Loop invariant:
    # At the beginning of each iteration, the `unmatched` set consists only of
    # students who are not currently matched with a hospital.

    # Initialization:
    # No students are matched with a hospital before entering the loop.
    unmatched = set(students)

    # Termination:
    # The algorithm terminates when every student has been matched to a hospital.
    while len(unmatched) > 0:
        for student in unmatched:
            student.apply_to_top_hospital()
        for hospital in hospitals:
            # Skip if no applications received this round
            if not hospital.applicants:
                continue
            hospital.sort_applicants()
            top_applicant = hospital.applicants[0]

            # Maintenance:
            # Students are removed from the unmatched set if a hospital
            # tentatively accepts them. If a hospital drops their match
            # for a student they like better, that student returns to the
            # unmatched set in preparation for the next iteration.

            if not hospital.match:
                # "Maybe" reply
                top_applicant.match = hospital
                hospital.match = top_applicant
                unmatched.remove(top_applicant)
                # "No" reply is implied
            else:
                # Check if hospital likes top applicant more than tentative match
                if hospital.student_ranking.index(
                    top_applicant
                ) < hospital.student_ranking.index(hospital.match):
                    # Old match gets the boot
                    unmatched.add(hospital.match)
                    hospital.match.match = None
                    # "Maybe" reply
                    hospital.match = top_applicant
                    top_applicant.match = hospital
                    unmatched.remove(top_applicant)
                # "No" reply is implied

            # Clear applications for next round
            hospital.applicants.clear()


def print_matches(students: list[Student]):
    """
    Print each resident and the hospital with which they have been matched.
    """

    for student in students:
        print(f"{student.name} {student.match.name}")


if __name__ == "__main__":
    main()
