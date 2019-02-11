from frameworks.factory import Framework


# class FrameworkSwitching(Framework):
#     def __init__(self, frameworks, framework_crossval, problem, curr_ref, model_list, *args, ** kwargs):
#         self.frameworks = frameworks
#         self.framework_crossval = framework_crossval
#         super().__init__(problem, curr_ref, model_list, *args, **kwargs)
#
#     def train(self, x, f, g, *args, **kwargs):
#
#         for s in self.frameworks:
#             get_framework(framework_id=self.framework_id,
#                           problem=self.problem,
#                           curr_ref=self.ref_dirs[self.cur_ref_no, :],
#                           model_list=self.model_list)
#
#     def predict(self, x, f, g, *args, **kwargs):
#         for i in range(0, self.problem.n_obj):
#             f[:, i] = self.model_list["f"+str(i+1)].predict(x)
#
#         if self.problem.n_constr > 0:
#             for i in range(0, self.problem.n_constr):
#                 g[:, i] = self.model_list["g" + str(i + 1)].predict(x)