from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
from requests.exceptions import ConnectionError, HTTPError
import rollbar

# Basic arguments. You should extend this function with the push features you
# want to use, or simply pass in a `PushMessage` object.


def send_push_message(token, title, message, extra=None):
    try:
        response = PushClient().publish(
            PushMessage(to=token,
                        data=extra,
                        title=title,
                        body=message,
                        sound='default',
                        badge=1
                        ))
        # print(response)
    except PushServerError as exc:
        # Encountered some likely formatting/validation error.
        rollbar.report_exc_info(
            extra_data={
                'token': token,
                'message': message,
                'extra': extra,
                'errors': exc.errors,
                'response_data': exc.response_data,
            })
        raise
    except (ConnectionError, HTTPError) as exc:
        # Encountered some Connection or HTTP error - retry a few times in
        # case it is transient.
        rollbar.report_exc_info(
            extra_data={'token': token, 'message': message, 'extra': extra})
        raise self.retry(exc=exc)

    try:
        # We got a response back, but we don't know whether it's an error yet.
        # This call raises errors so we can handle them with normal exception
        # flows.
        response.validate_response()
    except DeviceNotRegisteredError:
        # Mark the push token as inactive
        from notifications.models import PushToken
        PushToken.objects.filter(token=token).update(active=False)
    except PushTicketError as exc:
        # Encountered some other per-notification error.
        rollbar.report_exc_info(
            extra_data={
                'token': token,
                'message': message,
                'extra': extra,
                'push_response': exc.push_response._asdict(),
            })
        raise self.retry(exc=exc)


# Main function for testing during development
if __name__ == '__main__':
    token = 'ExponentPushToken[QdzwK-NUMCWMaVSyKnb8BC]'
    title = 'New Bird Memory! 🐦'
    message = 'A new bird memory has been captured!\nView it in your bird memories gallery.'

    send_push_message(token, title, message)
