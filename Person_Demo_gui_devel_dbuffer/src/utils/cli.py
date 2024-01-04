from threading import Thread
import numpy as np
from bullet import Bullet, Input, Numbers, VerticalPrompt, YesNo, colors, utils

class CLI:
    # Can read and write name_map freely.
    # Use db lock for otherwise
    def __init__(self, db):

        self.db = db
        self.thread = Thread(target=self.bullet_runner)
        self.started = False

    def start(self):
        if self.started:
            return

        self.started = True
        self.thread.start()
        return self

    def bullet_runner(self):
        while self.started:
            try:
                action_cli = Bullet(
                    prompt="\n What would you like to do : ",
                    choices=['Name user', 'List users',
                             'Delete user', 'Quit CLI'],
                    indent=0,
                    align=5,
                    margin=2,
                    bullet_color=colors.bright(colors.foreground["cyan"]),
                    background_color=colors.background["black"],
                    pad_right=5
                )

                user_id_cli = Numbers("Enter user id : ", type=int)
                name_user_cli = VerticalPrompt(
                    [user_id_cli, Input("Enter name for user : ", strip=True)])

                continue_cli = Input('Press any key to continue', default=' ')
                confirm_cli = YesNo('Are you sure ?')

                print('\033[0;0f\033[0J')
                action = action_cli.launch()

                if action == 'Quit CLI':
                    break
                elif action == 'Name user':
                    result = name_user_cli.launch()
                    confirm = confirm_cli.launch()

                    if confirm:
                        uid = result[0][1]
                        name = result[1][1]
                        self.db.lock()
                        self.db.name_map[uid] = name
                        self.db.unlock()

                    continue
                elif action == 'List users':
                    self.db.lock()

                    for i, v in enumerate(self.db.name_map):
                        if not np.all(self.db.face_mask[i] == 1):
                            continue
                        if not np.all(self.db.voice_mask[i] == 1):
                            continue
                        if np.all(self.db.pose_data[i] == -1):
                            continue
                        utils.forceWrite('%04d\t%s' %
                                         (i, v.decode('utf-8')), end='\n')

                    self.db.unlock()
                    continue_cli.launch()
                    continue
                elif action == 'Delete user':
                    result = user_id_cli.launch()
                    confirm = confirm_cli.launch()
                    if confirm:
                        self.db.lock()
                        self.db.delete_entry(result)
                        self.db.unlock()
                    continue
                else:
                    print(action)
            except BaseException:
                # If error in CLI, just reprint
                pass

    def stop(self):
        if self.started == False:
            return
        self.started = False
        self.thread.join()

    def __del__(self, **kwargs):
        self.stop()
