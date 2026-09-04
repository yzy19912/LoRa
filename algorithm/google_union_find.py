class UnionFind:
    def __init__(self, friends: list[str]):
        self.parent = {x: x for x in friends}
        self.groups = len(friends)

    def find(self, x: str) -> str:
        if self.parent.get(x) != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        parent_x, parent_y = self.find(x), self.find(y)
        if parent_x != parent_y:
            self.parent[parent_x] = parent_y
            self.groups -= 1

    def all_known(self):
        return self.groups == 1


def main():

    friends = ["Alice", "Bob", "Dan", "Erin"]

    uf = UnionFind(friends)

    log = [(1, "Alice", "Bob"), (2, "Bob", "Dan"), (3, "Dan", "Erin")]

    log.sort(key=lambda x: x[0])

    for ts, a, b in log:
        uf.union(a, b)
        if uf.all_known():
            print(f"all friends are known at {ts}")
            return
    print("this is impossible")


if __name__ == "__main__":
    main()
